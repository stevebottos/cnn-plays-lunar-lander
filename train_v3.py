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
from models import (
    Conv3DTransformerNet,
    Conv3dResNet,
    TemporalResNet,
    TemporalResNetGRU,
    TemporalMobileNetGRU,
    TinyCNN,
    TinyCNNv2,
    TinyCNNv2LSTM,
    TinyCNNv2Gated,
    TinyCNNv3,
)
from configs.config import TrainingConfig
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


NUM_STEPS_PER_ROLLOUT = 1024 * 8
FRAME_SIZE = 128
NUM_FRAMES_PER_BATCH = 16

STORAGE_DEVICE = "cpu"
INFERENCE_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class PPOManager:
    def __init__(self, config):
        self.config = config
        self.storage_device = torch.device(STORAGE_DEVICE)
        self.inference_device = torch.device(INFERENCE_DEVICE)
        self.agent = self._get_model().to(self.inference_device)

        self.optimizer = optim.Adam(self.agent.parameters(), lr=config.LEARNING_RATE)

        self.loaded_from_checkpoint = (
            hasattr(self, "checkpoint_optimizer_state")
            and self.checkpoint_optimizer_state is not None
        )
        if self.loaded_from_checkpoint:
            try:
                self.optimizer.load_state_dict(self.checkpoint_optimizer_state)
                print("✓ Restored optimizer state from checkpoint")
            except Exception as e:
                print(f"⚠ Could not restore optimizer state: {e}")
                self.loaded_from_checkpoint = False

        def lr_lambda(round_num):
            if self.loaded_from_checkpoint and round_num < 5:
                return 0.1 + 0.18 * round_num
            return 1.0

        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        self.env = self._get_env()

        if config.USE_MIXED_PRECISION and self.inference_device.type == "cuda":
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

        # Pre-allocated rollout storage
        self.obs = torch.zeros(
            NUM_STEPS_PER_ROLLOUT,
            1,
            NUM_FRAMES_PER_BATCH,
            FRAME_SIZE,
            FRAME_SIZE,
            dtype=torch.uint8,
            device=self.storage_device,
        )
        self.actions = torch.zeros(
            NUM_STEPS_PER_ROLLOUT, dtype=torch.int64, device=self.storage_device
        )
        self.log_probs = torch.zeros(
            NUM_STEPS_PER_ROLLOUT, dtype=torch.float32, device=self.storage_device
        )
        self.rewards = torch.zeros(
            NUM_STEPS_PER_ROLLOUT, dtype=torch.float32, device=self.storage_device
        )
        self.dones = torch.zeros(
            NUM_STEPS_PER_ROLLOUT, dtype=torch.float32, device=self.storage_device
        )
        self.values = torch.zeros(
            NUM_STEPS_PER_ROLLOUT, dtype=torch.float32, device=self.storage_device
        )
        self.advantages = torch.zeros(
            NUM_STEPS_PER_ROLLOUT, dtype=torch.float32, device=self.storage_device
        )
        self.returns = torch.zeros(
            NUM_STEPS_PER_ROLLOUT, dtype=torch.float32, device=self.storage_device
        )

    def _get_model(self):
        memops()

        self.checkpoint_optimizer_state = None

        if config.model_name == "TinyCNN":
            agent = TinyCNN(num_actions=4)
        elif config.model_name == "TinyCNNv2":
            agent = TinyCNNv2(num_actions=4)
        elif config.model_name == "TinyCNNv2LSTM":
            agent = TinyCNNv2LSTM(num_actions=4)
        elif config.model_name == "TinyCNNv2Gated":
            agent = TinyCNNv2Gated(num_actions=4)
        elif config.model_name == "TinyCNNv3":
            agent = TinyCNNv3(num_actions=4)
        elif config.model_name == "Conv3dResNet":
            agent = Conv3dResNet(num_actions=4)
        elif config.model_name == "TemporalResNet":
            agent = TemporalResNet(num_actions=4)
            try:
                checkpoint = torch.load(
                    "checkpoints/TEMPRES_STARTER.pt",
                    map_location=self.inference_device,
                )
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    agent.load_state_dict(checkpoint["model_state_dict"])
                    self.checkpoint_optimizer_state = checkpoint.get(
                        "optimizer_state_dict", None
                    )
                else:
                    agent.load_state_dict(checkpoint)
                    self.checkpoint_optimizer_state = None
                print(
                    "✓ Loaded TemporalResNet checkpoint from checkpoints/TEMPRES_STARTER.pt"
                )
            except FileNotFoundError:
                print("No TemporalResNet checkpoint found, starting fresh")
                self.checkpoint_optimizer_state = None
            except Exception as e:
                print(f"Error loading TemporalResNet checkpoint: {e}, starting fresh")
                self.checkpoint_optimizer_state = None
        elif config.model_name == "TemporalResNetGRU":
            agent = TemporalResNetGRU(num_actions=4)
            try:
                checkpoint = torch.load(
                    "checkpoints/GRU_STARTER.pt",
                    map_location=self.inference_device,
                )
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    agent.load_state_dict(checkpoint["model_state_dict"], strict=True)
                    self.checkpoint_optimizer_state = checkpoint.get(
                        "optimizer_state_dict", None
                    )
                else:
                    agent.load_state_dict(checkpoint, strict=True)
                    self.checkpoint_optimizer_state = None
                print(
                    "✓ Loaded TemporalResNetGRU checkpoint from checkpoints/GRU_STARTER.pt"
                )
            except FileNotFoundError:
                print("No checkpoint found, using fresh ImageNet weights")
                self.checkpoint_optimizer_state = None
            except Exception as e:
                print(f"Error loading checkpoint: {e}, using fresh ImageNet weights")
                self.checkpoint_optimizer_state = None
        elif config.model_name == "TemporalMobileNetGRU":
            agent = TemporalMobileNetGRU(num_actions=4)
            print("✓ Using pretrained MobileNetV3-Large from ImageNet")
        elif config.model_name == "Conv3DTransformerNet":
            agent = Conv3DTransformerNet(num_actions=4)
            try:
                state_dict = torch.load(
                    "checkpoints/TRANSFORMER_STARTER.pt",
                    map_location=self.inference_device,
                )
                agent.load_state_dict(state_dict)
                print(
                    "✓ Loaded transformer checkpoint from checkpoints/TRANSFORMER_STARTER.pt"
                )
            except FileNotFoundError:
                print("No transformer checkpoint found, starting fresh")
            except Exception as e:
                print(f"Error loading transformer checkpoint: {e}, starting fresh")
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
        env = ResizeObservation(env, (FRAME_SIZE, FRAME_SIZE))
        env = GrayscaleObservation(env, keep_dim=True)
        env = FrameStackObservation(env, NUM_FRAMES_PER_BATCH)
        return env

    def close(self):
        self.env.close()

    @torch.no_grad()
    def get_rollout(self):
        self.agent.eval()

        # Zero-out the storage tensors
        self.obs.zero_()
        self.actions.zero_()
        self.log_probs.zero_()
        self.rewards.zero_()
        self.dones.zero_()
        self.values.zero_()

        # Initialize environment and observation
        current_observation, info = self.env.reset()

        # Keep track of episode rewards for logging
        episode_rewards_list = []
        current_episode_rewards = 0

        step_idx = 0
        while step_idx < NUM_STEPS_PER_ROLLOUT:
            # Process current_observation and store directly into self.obs
            # The observation from the environment is a lazy frame stack (16, 128, 128, 1)
            # Permute to (1, 16, 128, 128) and store as float16 on CPU
            self.obs[step_idx] = (
                torch.from_numpy(current_observation)
                .permute(3, 0, 1, 2)
                .to(torch.uint8)
                .unsqueeze(1)
            )

            with torch.amp.autocast(
                "cuda", enabled=self.config.USE_MIXED_PRECISION, dtype=self.dtype
            ):
                agent_input = (
                    self.obs[step_idx].unsqueeze(0).to(self.inference_device) / 255.0
                )
                action_logits, value = self.agent(agent_input)
            action_dist = Categorical(logits=action_logits.float())
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)

            # Store action, log_prob, and value
            self.actions[step_idx] = action.to(self.storage_device)
            self.log_probs[step_idx] = log_prob.to(self.storage_device)
            self.values[step_idx] = value.squeeze().to(self.storage_device)

            # Execute action in the environment
            next_observation, reward, terminated, truncated, info = self.env.step(
                action.item()
            )
            done = terminated or truncated

            current_episode_rewards += reward

            # Store reward and done flag
            self.rewards[step_idx] = reward
            self.dones[step_idx] = float(done)  # 1.0 for done, 0.0 for not done

            # Update current_observation for the next step
            current_observation = next_observation

            if done:
                print(f"Episode finished. Total reward: {current_episode_rewards:.2f}")
                episode_rewards_list.append(current_episode_rewards)
                current_episode_rewards = 0  # Reset for next episode
                current_observation, info = self.env.reset()  # Reset environment

            step_idx += 1  # Increment step counter

        # --- GAE and Advantage Normalization ---
        last_done_flag = self.dones[NUM_STEPS_PER_ROLLOUT - 1].item()

        next_value_tensor = torch.tensor(0.0, device=self.inference_device)
        if last_done_flag == 0.0:  # If the last step was NOT terminal
            with torch.no_grad():
                agent_input_for_next_value = (
                    torch.from_numpy(current_observation)
                    .permute(3, 0, 1, 2)
                    .to(torch.uint8)
                    .unsqueeze(0)
                    .to(self.inference_device)
                    / 255.0
                )
                _, next_value_tensor = self.agent(agent_input_for_next_value)
                next_value_tensor = next_value_tensor.squeeze()

        self._calculate_gae(next_value_tensor, last_done_flag)
        self._normalize_advantages()

        return episode_rewards_list

    def _calculate_gae(self, next_value: torch.Tensor, last_done: float):
        """
        Calculates the Generalized Advantage Estimation (GAE) and returns for the rollout.
        This method operates on the pre-filled `self.rewards`, `self.values`, and `self.dones` tensors.
        """
        last_gae_lam = 0

        # We need to consider the value of the state after the last step of the rollout.
        # If the last step was a terminal state (done=1), the next value is 0.
        # Otherwise, it's the value estimated by the critic for that `next_observation`.
        next_value = next_value.reshape(1).to(self.storage_device)
        next_value_masked = next_value * (1.0 - last_done)

        values_with_next = torch.cat((self.values, next_value_masked), dim=0)

        for t in reversed(range(NUM_STEPS_PER_ROLLOUT)):
            mask = 1.0 - self.dones[t]
            delta = (
                self.rewards[t]
                + self.config.GAMMA * values_with_next[t + 1] * mask
                - values_with_next[t]
            )
            last_gae_lam = (
                delta + self.config.GAMMA * self.config.GAE_LAMBDA * mask * last_gae_lam
            )
            self.advantages[t] = last_gae_lam

        self.returns = (self.advantages + self.values).to(self.storage_device)

    def _normalize_advantages(self):
        """
        Normalizes the advantages tensor to have a mean of 0 and a standard deviation of 1.
        """
        mean = self.advantages.mean()
        std = self.advantages.std()
        self.advantages = (self.advantages - mean) / (std + 1e-8)

    def ppo_update(self):
        self.agent.train()

        # Data is already aggregated in self.obs, self.actions, etc.
        # Tensors remain on the storage_device (CPU) until needed for a mini-batch.

        # Calculate explained variance before training (using values from before GAE)
        explained_var = 1 - torch.var(self.returns - self.values) / (
            torch.var(self.returns) + 1e-8
        )

        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        total_ratio_mean = 0
        total_value_mean = 0
        total_clip_fraction = 0
        total_advantage_mean = 0
        total_advantage_std = 0
        total_approx_kl = 0
        num_updates = 0

        dataset_size = NUM_STEPS_PER_ROLLOUT
        indices = np.arange(dataset_size)

        # PPO Epochs loop
        for epoch in range(self.config.PPO_EPOCHS):
            np.random.shuffle(indices)

            # Mini-batch loop
            for start in range(0, dataset_size, self.config.BATCH_SIZE):
                end = start + self.config.BATCH_SIZE
                batch_indices = indices[start:end]

                # Get mini-batch data and move to the inference device
                states_mb = (
                    self.obs[batch_indices].to(self.inference_device, self.dtype)
                    / 255.0
                )
                actions_mb = self.actions[batch_indices].to(self.inference_device)
                old_log_probs_mb = self.log_probs[batch_indices].to(
                    self.inference_device
                )
                advantages_mb = self.advantages[batch_indices].to(self.inference_device)
                returns_mb = self.returns[batch_indices].to(self.inference_device)

                with torch.amp.autocast(
                    "cuda", enabled=self.config.USE_MIXED_PRECISION, dtype=self.dtype
                ):
                    action_logits, values = self.agent(states_mb)
                    action_dist = Categorical(logits=action_logits.float())
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

        # This block should be outside the epoch loop
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

    for round_num in range(config.NUM_ROUNDS):
        episode_rewards = (
            manager.get_rollout()
        )  # get_rollout now prepares all data and returns episode rewards

        # Update policy
        train_metrics = manager.ppo_update()

        # Calculate metrics for the collected rollout
        avg_reward = np.mean(episode_rewards) if episode_rewards else 0.0
        # avg_episode_length can be improved
        avg_episode_length = (
            manager.dones.sum().item() if manager.dones.sum().item() > 0 else 1
        )

        # Log metrics to MLflow
        mlflow.log_metric("Rewards/Average", avg_reward, step=round_num + 1)
        mlflow.log_metric("Episode/Length", avg_episode_length, step=round_num + 1)
        mlflow.log_metric(
            "Losses/Actor", train_metrics["actor_loss"], step=round_num + 1
        )
        mlflow.log_metric(
            "Losses/Critic", train_metrics["critic_loss"], step=round_num + 1
        )
        mlflow.log_metric(
            "Metrics/Entropy", train_metrics["entropy"], step=round_num + 1
        )
        mlflow.log_metric(
            "Metrics/Ratio_Mean", train_metrics["ratio_mean"], step=round_num + 1
        )
        mlflow.log_metric(
            "Metrics/Value_Mean", train_metrics["value_mean"], step=round_num + 1
        )
        mlflow.log_metric(
            "Metrics/Clip_Fraction", train_metrics["clip_fraction"], step=round_num + 1
        )
        mlflow.log_metric(
            "Metrics/Advantage_Mean",
            train_metrics["advantage_mean"],
            step=round_num + 1,
        )
        mlflow.log_metric(
            "Metrics/Advantage_Std", train_metrics["advantage_std"], step=round_num + 1
        )
        mlflow.log_metric(
            "Metrics/Approx_KL", train_metrics["approx_kl"], step=round_num + 1
        )
        mlflow.log_metric(
            "Metrics/Explained_Variance",
            train_metrics["explained_variance"],
            step=round_num + 1,
        )

        # Print progress
        if round_num % 10 == 0:
            lr_info = (
                f", LR={manager.scheduler.get_last_lr()[0]:.6f}"
                if manager.loaded_from_checkpoint and round_num < 10
                else ""
            )
            print(
                f"Round {round_num}: "
                f"Reward={avg_reward:.2f}, "
                f"Entropy={train_metrics['entropy']:.4f}, "
                f"Actor Loss={train_metrics['actor_loss']:.4f}, "
                f"Critic Loss={train_metrics['critic_loss']:.4f}, "
                f"Clip%={train_metrics['clip_fraction'] * 100:.1f}, "
                f"Adv(μ={train_metrics['advantage_mean']:.3f}, σ={train_metrics['advantage_std']:.3f}), "
                f"KL={train_metrics['approx_kl']:.3f}, ExpVar={train_metrics['explained_variance']:.3f})"
                f"{lr_info}"
            )

        # Create log entry for the current round and append to jsonl file
        current_log = {
            "round": round_num,
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

        if round_num % 50 == 0:
            checkpoint_path = checkpoints_out / f"{str(round_num).zfill(7)}.pt"
            torch.save(
                {
                    "model_state_dict": manager.agent.state_dict(),
                    "optimizer_state_dict": manager.optimizer.state_dict(),
                    "round": round_num,
                },
                str(checkpoint_path),
            )

        manager.scheduler.step()

    manager.close()
    mlflow.end_run()
    tracemalloc.stop()
