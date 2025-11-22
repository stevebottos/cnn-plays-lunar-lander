# type: ignore
from pathlib import Path
import json
import gymnasium as gym
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import mlflow
import numpy as np
import argparse
import yaml
import os
import gc
from datetime import datetime
from models import Conv3DTransformerNet, Conv3dResNet, TinyCNN, TinyCNNv2, TinyCNNv3
from configs.config import TrainingConfig
from collections import namedtuple, deque
import tracemalloc
from gymnasium.wrappers import (
    AddRenderObservation,
    ResizeObservation,
    GrayscaleObservation,
    FrameStackObservation,
)


def load_config(config_path: str) -> TrainingConfig:
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    return TrainingConfig(**config_dict), config_dict


def memops():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_flash_sdp(True)


def calculate_gae(rewards, values, next_value, dones, gamma, gae_lambda, device):
    advantages = []
    gae = 0
    rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
    if isinstance(next_value, (int, float)):
        next_value = torch.tensor(next_value, dtype=torch.float32, device=device)
    elif next_value.dim() > 0:
        next_value = next_value.squeeze()

    values_list = torch.cat([values, next_value.unsqueeze(0)])
    dones = torch.tensor(dones, dtype=torch.float32, device=device)

    for t in reversed(range(len(rewards))):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * values_list[t + 1] * mask - values_list[t]
        gae = delta + gamma * gae_lambda * mask * gae
        advantages.insert(0, gae.detach())  # CRITICAL: Detach to break gradient chain

    advantages = torch.stack(advantages)
    returns = (advantages + values).detach()

    del rewards, values_list, next_value, dones, gae, delta
    return advantages, returns


class PPOManager:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agent = self._get_model().to(self.device)

        try:
            state_dict = torch.load("checkpoints/STARTER.pt")
            self.agent.load_state_dict(state_dict)
        except:
            print("Unable to load from checkpoint.")

        self.optimizer = optim.Adam(self.agent.parameters(), lr=config.LEARNING_RATE)
        self.env = self._get_env()

        if config.USE_MIXED_PRECISION and self.device.type == "cuda":
            if torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
            else:
                dtype = torch.float16
            scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))
        else:
            dtype = torch.float32
            scaler = None

        self.dtype = dtype
        self.scaler = scaler
        self.episode_count = 0

        # Running statistics for return normalization
        self.return_rms_mean = 0.0
        self.return_rms_var = 1.0
        self.return_rms_count = 1e-4

    def _get_model(self):
        if config.model_name == "TinyCNN":
            agent = TinyCNN(num_actions=4)
        elif config.model_name == "TinyCNNv2":
            agent = TinyCNNv2(num_actions=4)
        elif config.model_name == "TinyCNNv3":
            agent = TinyCNNv3(num_actions=4)
        elif config.model_name == "Conv3dResNet":
            agent = Conv3dResNet(num_actions=4)
        elif config.model_name == "Conv3DTransformerNet":
            agent = Conv3DTransformerNet(num_actions=4)
            memops()
        else:
            raise ValueError(f"Unknown model_name: {config.model_name}")

        if config.USE_TORCH_COMPILE:
            try:
                agent = torch.compile(agent, mode="reduce-overhead")
            except Exception:
                pass
        return agent

    def _get_env(self):
        env = gym.make(
            "LunarLander-v3",
            render_mode="rgb_array",
            max_episode_steps=250,
        )
        env = AddRenderObservation(env)
        env = ResizeObservation(env, (128, 128))
        env = GrayscaleObservation(env, keep_dim=True)
        env = FrameStackObservation(env, 16)
        return env

    def close(self):
        self.env.close()

    def update_return_rms(self, returns):
        """Update running mean and std for return normalization"""
        batch_mean = returns.mean().item()
        batch_var = returns.var().item()
        batch_count = len(returns)

        delta = batch_mean - self.return_rms_mean
        total_count = self.return_rms_count + batch_count

        new_mean = self.return_rms_mean + delta * batch_count / total_count
        m_a = self.return_rms_var * self.return_rms_count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.return_rms_count * batch_count / total_count
        new_var = M2 / total_count

        self.return_rms_mean = new_mean
        self.return_rms_var = new_var
        self.return_rms_count = total_count

    @torch.no_grad()
    def get_rollout(self):
        self.agent.eval()
        episode = namedtuple(
            "Episode",
            [
                "states",
                "actions",
                "log_probs",
                "rewards",
                "advantages",
                "returns",
            ],
        )
        rollout_buffer = []

        for i in range(self.config.COLLECT_EPISODES):
            episode_states = []
            episode_actions = []
            episode_log_probs = []
            episode_values = []
            episode_rewards = []
            episode_dones = []

            observation, info = self.env.reset()
            terminated, truncated = False, False
            steps = 0

            while not terminated and not truncated:
                image_tensor = (
                    torch.from_numpy(observation)
                    .permute(3, 0, 1, 2)
                    .float()
                    .unsqueeze(0)
                )
                input_tensor = image_tensor.to(self.device) / 255.0
                episode_states.append(input_tensor.detach().cpu())

                with torch.amp.autocast(
                    "cuda", enabled=config.USE_MIXED_PRECISION, dtype=self.dtype
                ):
                    action_logits, value = self.agent(input_tensor)

                action_dist = Categorical(logits=action_logits.float())
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action)

                episode_actions.append(action.detach().cpu())
                episode_log_probs.append(log_prob.detach().cpu())
                episode_values.append(value.squeeze(-1).detach().cpu())

                observation, reward, terminated, truncated, info = self.env.step(
                    action.item()
                )
                episode_rewards.append(reward)
                episode_dones.append(terminated)
                steps += 1

            if episode_rewards:
                if terminated:
                    next_value = torch.tensor(0.0, device="cpu")
                else:
                    image_tensor = (
                        torch.from_numpy(observation)
                        .permute(3, 0, 1, 2)
                        .float()
                        .unsqueeze(0)
                    )
                    input_tensor = image_tensor.to(self.device) / 255.0
                    with torch.amp.autocast(
                        "cuda", enabled=config.USE_MIXED_PRECISION, dtype=self.dtype
                    ):
                        _, next_value = self.agent(input_tensor)
                        next_value = next_value.squeeze(-1).detach().cpu()
                    del image_tensor, input_tensor

                values_tensor = torch.cat(episode_values)
                advantages, returns = calculate_gae(
                    episode_rewards,
                    values_tensor,
                    next_value if not terminated else 0.0,
                    episode_dones,
                    config.GAMMA,
                    config.GAE_LAMBDA,
                    "cpu",
                )

                print(f"Episode reward: {sum(episode_rewards):.2f}")

                rollout_buffer.append(
                    episode(
                        states=episode_states,
                        actions=episode_actions,
                        log_probs=episode_log_probs,
                        rewards=episode_rewards,
                        advantages=advantages,
                        returns=returns,
                    )
                )

                del (
                    episode_states,
                    episode_actions,
                    episode_log_probs,
                    episode_values,
                    episode_rewards,
                    episode_dones,
                    values_tensor,
                    advantages,
                    returns,
                    next_value,
                )

                self.episode_count += 1

        return rollout_buffer

    def ppo_update(self, rollout_buffer):
        self.agent.train()

        # Aggregate data from rollout buffer
        states_batch = torch.cat(
            [torch.stack(ep.states) for ep in rollout_buffer], dim=0
        )
        actions_batch = torch.cat(
            [torch.stack(ep.actions) for ep in rollout_buffer], dim=0
        )
        old_log_probs_batch = torch.cat(
            [torch.stack(ep.log_probs) for ep in rollout_buffer], dim=0
        )
        advantages_batch = torch.cat([ep.advantages for ep in rollout_buffer], dim=0)
        returns_batch = torch.cat([ep.returns for ep in rollout_buffer], dim=0)

        # Store raw advantages for normalization in the epoch loop
        advantages_raw = advantages_batch.clone()

        # Update running statistics and normalize returns
        if self.config.NORMALIZE_RETURNS:
            self.update_return_rms(returns_batch)
            mean = torch.tensor(self.return_rms_mean)
            std = torch.sqrt(torch.tensor(self.return_rms_var))
            returns_batch_norm = (returns_batch - mean) / (std + 1e-8)
        else:
            returns_batch_norm = returns_batch

        if self.config.NORMALIZE_ADVANTAGES:
            # Standard PPO: zero mean, unit variance
            adv_mean = advantages_raw.mean()
            adv_std = advantages_raw.std()
            advantages_batch = (advantages_raw - adv_mean) / (adv_std + 1e-8)
        else:
            # Just use raw advantages (not recommended, but if you insist)
            advantages_batch = advantages_raw

        # Calculate explained variance
        values_batch = returns_batch - advantages_batch
        explained_var = 1 - torch.var(returns_batch - values_batch) / (
            torch.var(returns_batch) + 1e-8
        )
        # Removed direct mlflow.log_metric for Explained_Variance, now returned and logged in main loop

        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        total_ratio_mean = 0
        total_value_mean = 0
        total_clip_fraction = 0
        total_advantage_mean = 0
        total_advantage_std = 0
        total_approx_kl = 0  # ADDED: Initialization for approximate KL divergence
        num_updates = 0

        dataset_size = len(states_batch)
        indices = np.arange(dataset_size)

        # PPO Epochs loop
        for epoch in range(self.config.PPO_EPOCHS):
            np.random.shuffle(indices)

            # Mini-batch loop
            for start in range(0, dataset_size, self.config.BATCH_SIZE):
                end = start + self.config.BATCH_SIZE
                batch_indices = indices[start:end]

                # Get mini-batch data
                states_mb = states_batch[batch_indices].to(self.device).squeeze(1)
                actions_mb = actions_batch[batch_indices].to(self.device)
                old_log_probs_mb = old_log_probs_batch[batch_indices].to(self.device)
                advantages_mb = advantages_batch[batch_indices].to(self.device)
                returns_mb = returns_batch_norm[batch_indices].to(self.device)

                with torch.amp.autocast(
                    "cuda", enabled=self.config.USE_MIXED_PRECISION, dtype=self.dtype
                ):
                    action_logits, values = self.agent(states_mb)
                    action_dist = Categorical(logits=action_logits)
                    log_probs = action_dist.log_prob(actions_mb)
                    entropy = action_dist.entropy().mean()

                    # Actor loss with PPO clipping
                    ratio = torch.exp(log_probs - old_log_probs_mb)

                    # Calculate Approximate KL Divergence
                    with torch.no_grad():
                        approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()
                    total_approx_kl += approx_kl
                    surr1 = ratio * advantages_mb
                    surr2 = (
                        torch.clamp(
                            ratio,
                            1.0 - self.config.CLIP_EPSILON,
                            1.0 + self.config.CLIP_EPSILON,
                        )
                        * advantages_mb
                    )
                    actor_loss = -torch.min(surr1, surr2).mean()

                    # Track clipping statistics
                    clip_fraction = (
                        ((ratio - 1.0).abs() > self.config.CLIP_EPSILON).float().mean()
                    )

                    # Critic loss
                    critic_loss = F.mse_loss(values.squeeze(-1), returns_mb)

                    # Total loss
                    total_loss = (
                        actor_loss
                        + self.config.VALUE_COEFF * critic_loss
                        - self.config.ENTROPY_COEFF * entropy
                    )

                self.optimizer.zero_grad()
                if self.scaler:
                    self.scaler.scale(total_loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 1.0)
                    self.optimizer.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.item()
                total_ratio_mean += ratio.mean().item()
                total_value_mean += values.mean().item()
                total_clip_fraction += clip_fraction.item()
                total_advantage_mean += advantages_mb.mean().item()
                total_advantage_std += advantages_mb.std().item()
                num_updates += 1

                del states_mb, actions_mb, old_log_probs_mb, advantages_mb, returns_mb
                del action_logits, values, action_dist, log_probs, entropy
                del ratio, surr1, surr2, actor_loss, critic_loss, total_loss

        del (
            states_batch,
            actions_batch,
            old_log_probs_batch,
            advantages_batch,
            advantages_raw,
            returns_batch,
            returns_batch_norm,
            indices,
        )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "actor_loss": total_actor_loss / num_updates,
            "critic_loss": total_critic_loss / num_updates,
            "entropy": total_entropy / num_updates,
            "ratio_mean": total_ratio_mean / num_updates,
            "value_mean": total_value_mean / num_updates,
            "clip_fraction": total_clip_fraction / num_updates,
            "advantage_mean": total_advantage_mean / num_updates,
            "advantage_std": total_advantage_std / num_updates,
            "approx_kl": total_approx_kl / num_updates,
            "explained_variance": explained_var.item(),
        }


if __name__ == "__main__":
    tracemalloc.start()
    parser = argparse.ArgumentParser(description="PPO Training Script")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    args = parser.parse_args()

    config, config_as_dict = load_config(args.config)
    config_name = os.path.splitext(os.path.basename(args.config))[0]

    checkpoints_out = Path("checkpoints")
    checkpoints_out.mkdir(parents=True, exist_ok=True)

    # Recreate run.jsonl at the start of each run
    jsonl_file_path = "run.jsonl"
    if os.path.exists(jsonl_file_path):
        os.remove(jsonl_file_path)
        print(f"Recreated {jsonl_file_path} for new run.")

    # Setup MLflow
    mlflow.set_experiment(config_name)
    run_name = f"{config.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run = mlflow.start_run(run_name=run_name)
    mlflow.log_params(config_as_dict)
    print(f"MLflow experiment '{config_name}' started with run '{run_name}'.")

    manager = PPOManager(config)

    for round in range(config.NUM_ROUNDS):
        rollout_buffer = manager.get_rollout()

        # Calculate metrics BEFORE ppo_update
        avg_reward = np.mean([np.sum(ep.rewards) for ep in rollout_buffer])
        avg_episode_length = np.mean([len(ep.rewards) for ep in rollout_buffer])

        # Update policy
        train_metrics = manager.ppo_update(rollout_buffer)

        del rollout_buffer
        gc.collect()

        # Log metrics to MLflow
        mlflow.log_metric("Rewards/Average", avg_reward, step=round + 1)
        mlflow.log_metric("Episode/Length", avg_episode_length, step=round + 1)
        mlflow.log_metric("Losses/Actor", train_metrics["actor_loss"], step=round + 1)
        mlflow.log_metric("Losses/Critic", train_metrics["critic_loss"], step=round + 1)
        mlflow.log_metric("Metrics/Entropy", train_metrics["entropy"], step=round + 1)
        mlflow.log_metric(
            "Metrics/Ratio_Mean", train_metrics["ratio_mean"], step=round + 1
        )
        mlflow.log_metric(
            "Metrics/Value_Mean", train_metrics["value_mean"], step=round + 1
        )
        mlflow.log_metric(
            "Metrics/Clip_Fraction", train_metrics["clip_fraction"], step=round + 1
        )
        mlflow.log_metric(
            "Metrics/Advantage_Mean", train_metrics["advantage_mean"], step=round + 1
        )
        mlflow.log_metric(
            "Metrics/Advantage_Std", train_metrics["advantage_std"], step=round + 1
        )
        mlflow.log_metric(
            "Metrics/Approx_KL", train_metrics["approx_kl"], step=round + 1
        )
        mlflow.log_metric(
            "Metrics/Explained_Variance",
            train_metrics["explained_variance"],
            step=round + 1,
        )

        # Print progress
        if round % 10 == 0:
            print(
                f"Round {round}: "
                f"Reward={avg_reward:.2f}, "
                f"Entropy={train_metrics['entropy']:.4f}, "
                f"Actor Loss={train_metrics['actor_loss']:.4f}, "
                f"Critic Loss={train_metrics['critic_loss']:.4f}, "
                f"Clip%={train_metrics['clip_fraction'] * 100:.1f}, "
                f"Adv(μ={train_metrics['advantage_mean']:.3f}, σ={train_metrics['advantage_std']:.3f}), "
                f"KL={train_metrics['approx_kl']:.3f}, ExpVar={train_metrics['explained_variance']:.3f})"
            )

        # Create log entry for the current round and append to jsonl file
        current_log = {
            "round": round,
            "Rewards/Average": avg_reward,
            "Episode/Length": avg_episode_length,
            "Losses/Actor": train_metrics["actor_loss"],
            "Losses/Critic": train_metrics["critic_loss"],
            "Metrics/Entropy": train_metrics["entropy"],
            "Metrics/Ratio_Mean": train_metrics["ratio_mean"],
            "Metrics/Value_Mean": train_metrics["value_mean"],
            "Metrics/Clip_Fraction": train_metrics["clip_fraction"],
            "Metrics/Advantage_Mean": train_metrics["advantage_mean"],
            "Metrics/Advantage_Std": train_metrics["advantage_std"],
            "Metrics/Approx_KL": train_metrics["approx_kl"],
            "Metrics/Explained_Variance": train_metrics["explained_variance"],
        }
        with open("run.jsonl", "a") as f:
            f.write(json.dumps(current_log) + "\n")

        if round % 50 == 0:
            checkpoint_path = checkpoints_out / f"{str(round).zfill(7)}.pt"
            torch.save(manager.agent.state_dict(), str(checkpoint_path))

    manager.close()
    mlflow.end_run()
    tracemalloc.stop()
