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


# Parallel environment configuration
NUM_ENVS = 8  # Number of parallel environments

FRAME_SIZE = 128
NUM_FRAMES_PER_BATCH = 16

STORAGE_DEVICE = "cpu"
INFERENCE_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def make_env(max_episode_steps=250):
    """Factory function to create a single wrapped environment"""
    env = gym.make(
        "LunarLander-v3",
        render_mode="rgb_array",
        max_episode_steps=max_episode_steps,
    )
    env = AddRenderObservation(env)
    env = ResizeObservation(env, (FRAME_SIZE, FRAME_SIZE))
    env = GrayscaleObservation(env, keep_dim=True)
    env = FrameStackObservation(env, NUM_FRAMES_PER_BATCH)
    return env


class PPOManager:
    def __init__(self, config):
        self.config = config
        self.num_steps_per_rollout = config.NUM_STEPS_PER_ROLLOUT
        self.storage_device = torch.device(STORAGE_DEVICE)
        self.inference_device = torch.device(INFERENCE_DEVICE)
        self.agent = self._get_model().to(self.inference_device)

        # Setup optimizer with optional separate learning rate for GRU
        if (
            config.model_name == "TemporalResNetGRU"
            and config.GRU_LEARNING_RATE is not None
        ):
            # Use different learning rates for pretrained backbone vs trained-from-scratch GRU
            backbone_params = list(self.agent.backbone.parameters())
            gru_and_heads_params = (
                list(self.agent.gru.parameters())
                + list(self.agent.norm.parameters())
                + list(self.agent.actor.parameters())
                + list(self.agent.critic.parameters())
            )
            self.optimizer = optim.Adam(
                [
                    {"params": backbone_params, "lr": config.LEARNING_RATE},
                    {"params": gru_and_heads_params, "lr": config.GRU_LEARNING_RATE},
                ]
            )
            print(
                f"Using separate learning rates: Backbone LR={config.LEARNING_RATE}, GRU+Heads LR={config.GRU_LEARNING_RATE}"
            )
        else:
            self.optimizer = optim.Adam(
                self.agent.parameters(), lr=config.LEARNING_RATE
            )

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

        # No warmup - constant learning rate throughout
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lambda round_num: 1.0)

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

        # Pre-allocated rollout storage with pinned memory for faster CPU→GPU transfers
        pin = self.storage_device.type == "cpu"
        self.obs = torch.zeros(
            self.num_steps_per_rollout,
            NUM_ENVS,
            1,
            NUM_FRAMES_PER_BATCH,
            FRAME_SIZE,
            FRAME_SIZE,
            dtype=torch.uint8,
            device=self.storage_device,
            pin_memory=pin,
        )
        self.actions = torch.zeros(
            self.num_steps_per_rollout,
            NUM_ENVS,
            dtype=torch.int64,
            device=self.storage_device,
            pin_memory=pin,
        )
        self.log_probs = torch.zeros(
            self.num_steps_per_rollout,
            NUM_ENVS,
            dtype=torch.float32,
            device=self.storage_device,
            pin_memory=pin,
        )
        self.rewards = torch.zeros(
            self.num_steps_per_rollout,
            NUM_ENVS,
            dtype=torch.float32,
            device=self.storage_device,
            pin_memory=pin,
        )
        self.dones = torch.zeros(
            self.num_steps_per_rollout,
            NUM_ENVS,
            dtype=torch.float32,
            device=self.storage_device,
            pin_memory=pin,
        )
        self.values = torch.zeros(
            self.num_steps_per_rollout,
            NUM_ENVS,
            dtype=torch.float32,
            device=self.storage_device,
            pin_memory=pin,
        )
        self.advantages = torch.zeros(
            self.num_steps_per_rollout,
            NUM_ENVS,
            dtype=torch.float32,
            device=self.storage_device,
            pin_memory=pin,
        )
        self.returns = torch.zeros(
            self.num_steps_per_rollout,
            NUM_ENVS,
            dtype=torch.float32,
            device=self.storage_device,
            pin_memory=pin,
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

            # Compile BEFORE loading checkpoint to match saved state_dict structure
            if config.USE_TORCH_COMPILE:
                try:
                    agent = torch.compile(agent, mode="reduce-overhead")
                    print("✓ Model compiled with torch.compile")
                except Exception as e:
                    print(f"⚠ torch.compile failed: {e}")

            # Load checkpoint if specified in config
            if config.CHECKPOINT_PATH:
                try:
                    checkpoint = torch.load(
                        config.CHECKPOINT_PATH,
                        map_location=self.inference_device,
                    )
                    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                        result = agent.load_state_dict(
                            checkpoint["model_state_dict"], strict=True
                        )
                        self.checkpoint_optimizer_state = checkpoint.get(
                            "optimizer_state_dict", None
                        )
                        print(f"✓ Loaded TemporalResNetGRU checkpoint from {config.CHECKPOINT_PATH}")
                        print(f"  Missing keys: {result.missing_keys}")
                        print(f"  Unexpected keys: {result.unexpected_keys}")
                    else:
                        result = agent.load_state_dict(checkpoint, strict=True)
                        self.checkpoint_optimizer_state = None
                        print(f"✓ Loaded TemporalResNetGRU checkpoint from {config.CHECKPOINT_PATH}")
                        print(f"  Missing keys: {result.missing_keys}")
                        print(f"  Unexpected keys: {result.unexpected_keys}")
                except FileNotFoundError:
                    print(f"⚠ Checkpoint not found at {config.CHECKPOINT_PATH}, using fresh ImageNet weights")
                    self.checkpoint_optimizer_state = None
                except Exception as e:
                    print(f"⚠ Error loading checkpoint from {config.CHECKPOINT_PATH}: {e}")
                    print("  Using fresh ImageNet weights")
                    self.checkpoint_optimizer_state = None
            else:
                print("No checkpoint specified, using fresh ImageNet weights")
                self.checkpoint_optimizer_state = None

            return agent
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
        # Create asynchronous vectorized environment (runs in parallel subprocesses)
        max_steps = self.config.MAX_EPISODE_STEPS
        envs = gym.vector.AsyncVectorEnv([lambda: make_env(max_steps) for _ in range(NUM_ENVS)])
        return envs

    def close(self):
        self.env.close()

    @torch.no_grad()
    def get_rollout(self):
        self.agent.eval()

        self.obs.zero_()
        self.actions.zero_()
        self.log_probs.zero_()
        self.rewards.zero_()
        self.dones.zero_()
        self.values.zero_()

        current_observations, infos = self.env.reset()

        episode_rewards_list = []
        episode_lengths_list = []
        current_episode_rewards = np.zeros(NUM_ENVS)
        current_episode_lengths = np.zeros(NUM_ENVS, dtype=np.int32)

        for step_idx in range(self.num_steps_per_rollout):
            obs_batch = (
                torch.from_numpy(current_observations)
                .permute(0, 4, 1, 2, 3)
                .to(torch.uint8)
            )
            self.obs[step_idx] = obs_batch

            with torch.amp.autocast(
                "cuda", enabled=self.config.USE_MIXED_PRECISION, dtype=self.dtype
            ):
                agent_input = self.obs[step_idx].to(self.inference_device) / 255.0
                action_logits, values = self.agent(agent_input)

            action_dist = Categorical(logits=action_logits.float())
            actions = action_dist.sample()
            log_probs = action_dist.log_prob(actions)

            self.actions[step_idx] = actions.to(self.storage_device)
            self.log_probs[step_idx] = log_probs.to(self.storage_device)
            self.values[step_idx] = values.squeeze(-1).to(self.storage_device)

            next_observations, rewards, terminateds, truncateds, infos = self.env.step(
                actions.cpu().numpy()
            )
            dones = np.logical_or(terminateds, truncateds)

            current_episode_rewards += rewards
            current_episode_lengths += 1

            self.rewards[step_idx] = torch.from_numpy(rewards).to(self.storage_device)
            self.dones[step_idx] = torch.from_numpy(dones.astype(np.float32)).to(
                self.storage_device
            )

            current_observations = next_observations

            for env_idx in range(NUM_ENVS):
                if dones[env_idx]:
                    print(
                        f"Env {env_idx} - Episode finished. Total reward: {current_episode_rewards[env_idx]:.2f}, Length: {current_episode_lengths[env_idx]}"
                    )
                    episode_rewards_list.append(current_episode_rewards[env_idx])
                    episode_lengths_list.append(current_episode_lengths[env_idx])
                    current_episode_rewards[env_idx] = 0
                    current_episode_lengths[env_idx] = 0

        with torch.no_grad():
            agent_input_for_next_value = (
                torch.from_numpy(current_observations)
                .permute(0, 4, 1, 2, 3)
                .to(torch.uint8)
                .to(self.inference_device)
                / 255.0
            )
            _, next_values = self.agent(agent_input_for_next_value)
            next_values = next_values.squeeze(-1).to(self.storage_device)

        last_dones = self.dones[self.num_steps_per_rollout - 1]
        next_values = next_values * (1.0 - last_dones)

        self._calculate_gae(next_values)
        self._normalize_advantages()

        # Debug: Check value statistics
        print(f"\nRollout stats:")
        print(
            f"  Values - min: {self.values.min():.2f}, max: {self.values.max():.2f}, mean: {self.values.mean():.2f}"
        )
        print(
            f"  Returns - min: {self.returns.min():.2f}, max: {self.returns.max():.2f}, mean: {self.returns.mean():.2f}"
        )
        print(
            f"  Rewards - min: {self.rewards.min():.2f}, max: {self.rewards.max():.2f}, mean: {self.rewards.mean():.2f}"
        )

        return episode_rewards_list, episode_lengths_list

    def _calculate_gae(self, next_values: torch.Tensor):
        """
        Calculates the Generalized Advantage Estimation (GAE) and returns for the rollout.
        This method operates on the pre-filled `self.rewards`, `self.values`, and `self.dones` tensors.
        """
        last_gae_lam = torch.zeros(NUM_ENVS, device=self.storage_device)
        values_with_next = torch.cat((self.values, next_values.unsqueeze(0)), dim=0)

        for t in reversed(range(self.num_steps_per_rollout)):
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

        # Flatten (num_steps_per_rollout, NUM_ENVS, ...) to (num_steps_per_rollout * NUM_ENVS, ...)
        obs_flat = self.obs.reshape(-1, 1, NUM_FRAMES_PER_BATCH, FRAME_SIZE, FRAME_SIZE)
        actions_flat = self.actions.reshape(-1)
        log_probs_flat = self.log_probs.reshape(-1)
        advantages_flat = self.advantages.reshape(-1)
        returns_flat = self.returns.reshape(-1)
        values_flat = self.values.reshape(-1)

        # Calculate explained variance before training
        explained_var = 1 - torch.var(returns_flat - values_flat) / (
            torch.var(returns_flat) + 1e-8
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

        dataset_size = self.num_steps_per_rollout * NUM_ENVS
        indices = np.arange(dataset_size)

        # PPO Epochs loop
        for epoch in range(self.config.PPO_EPOCHS):
            np.random.shuffle(indices)

            # Mini-batch loop
            num_batches = (
                dataset_size + self.config.BATCH_SIZE - 1
            ) // self.config.BATCH_SIZE
            for batch_idx, start in enumerate(
                range(0, dataset_size, self.config.BATCH_SIZE)
            ):
                end = start + self.config.BATCH_SIZE
                batch_indices = indices[start:end]

                # Simple progress bar
                progress = (batch_idx + 1) / num_batches
                bar_length = 60
                filled = int(bar_length * progress)
                bar = "=" * filled + "-" * (bar_length - filled)
                print(
                    f"\rEpoch {epoch + 1}/{self.config.PPO_EPOCHS} [{bar}] {batch_idx + 1}/{num_batches}",
                    end="",
                    flush=True,
                )

                # Get mini-batch data from flattened tensors
                states_mb = (
                    obs_flat[batch_indices].to(self.inference_device, self.dtype)
                    / 255.0
                )
                actions_mb = actions_flat[batch_indices].to(self.inference_device)
                old_log_probs_mb = log_probs_flat[batch_indices].to(
                    self.inference_device
                )
                advantages_mb = advantages_flat[batch_indices].to(self.inference_device)
                returns_mb = returns_flat[batch_indices].to(self.inference_device)

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

        print()  # New line after progress bar

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
        episode_rewards, episode_lengths = manager.get_rollout()

        # Update policy
        train_metrics = manager.ppo_update()

        # Calculate metrics for the collected rollout
        avg_reward = np.mean(episode_rewards) if episode_rewards else 0.0
        avg_episode_length = np.mean(episode_lengths) if episode_lengths else 0.0

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
            print(
                f"Round {round_num}: "
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
