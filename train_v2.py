# type: ignore
import gymnasium as gym
from gymnasium.wrappers import (
    AddRenderObservation,
    ResizeObservation,
    GrayscaleObservation,
    FrameStackObservation,
)
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import mlflow
import mlflow.tracking
import numpy as np
import cv2
import sys
import argparse
import yaml
import os
from datetime import datetime
import tempfile

from models import Conv3DTransformerNet, Conv3dResNet, TinyCNN, TinyCNNv2
from configs.config import TrainingConfig
from collections import namedtuple


def load_config(config_path: str) -> TrainingConfig:
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    return TrainingConfig(**config_dict)


def calculate_gae(rewards, values, next_value, gamma, gae_lambda, device):
    advantages = []
    gae = 0
    rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
    if isinstance(next_value, (int, float)):
        next_value = torch.tensor(next_value, dtype=torch.float32, device=device)
    elif next_value.dim() > 0:
        next_value = next_value.squeeze()
    values_list = torch.cat([values, next_value.unsqueeze(0)])
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values_list[t + 1] - values_list[t]
        gae = delta + gamma * gae_lambda * gae
        advantages.insert(0, gae)
    advantages = torch.stack(advantages)
    returns = advantages + values
    return advantages, returns


class PPOManager:
    def __init__(self, config):
        self.config = config

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agent = self._get_model().to(self.device)
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

    def _get_model(self):
        if config.model_name == "TinyCNN":
            agent = TinyCNN(num_actions=4)
        elif config.model_name == "TinyCNNv2":
            agent = TinyCNNv2(num_actions=4)
        elif config.model_name == "Conv3dResNet":
            agent = Conv3dResNet(num_actions=4)
        # elif config.model_name == "Conv3DTransformerNet":
        #     agent = Conv3DTransformerNet(num_actions=4, num_frames=config.NUM_FRAMES)
        else:
            raise ValueError(f"Unknown model_name: {config.model_name}")

        # TODO: rework this, only works for transformer
        if config.USE_TORCH_COMPILE:
            try:
                agent = torch.compile(agent, mode="reduce-overhead")
            except Exception:
                pass  # Keep silent if compilation fails

        return agent

    def _get_env(self):
        env = gym.make(
            config.env_name,
            render_mode="rgb_array",
            max_episode_steps=config.MAX_EPISODE_STEPS,
        )
        # This is always 4 for lunar lander so who cares right now
        # NUM_ACTIONS = env.action_space.n
        env = AddRenderObservation(env)
        env = ResizeObservation(env, shape=(config.IMAGE_SIZE, config.IMAGE_SIZE))
        env = GrayscaleObservation(env, keep_dim=True)
        env = FrameStackObservation(env, config.NUM_FRAMES)

        return env

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
            observation, info = self.env.reset()
            terminated, truncated = False, False
            steps = 0

            while not terminated and not truncated:
                image_tensor = torch.from_numpy(observation).permute(3, 0, 1, 2).float()
                input_tensor = image_tensor.to(self.device) / 255.0
                episode_states.append(input_tensor.cpu())
                with torch.amp.autocast(
                    "cuda", enabled=config.USE_MIXED_PRECISION, dtype=self.dtype
                ):
                    action_logits, value = self.agent(input_tensor.unsqueeze(0))

                action_dist = Categorical(logits=action_logits.float())
                action = action_dist.sample()
                episode_actions.append(action.detach().cpu())
                episode_log_probs.append(action_dist.log_prob(action).detach().cpu())
                episode_values.append(value.squeeze(-1).detach().cpu())
                observation, reward, terminated, truncated, info = self.env.step(
                    action.item()
                )
                clipped_reward = np.clip(
                    reward, -config.REWARD_CLIP, config.REWARD_CLIP
                )
                episode_rewards.append(clipped_reward)
                steps += 1

            if episode_rewards:
                if terminated:
                    next_value = torch.tensor(0.0, device=self.device)
                else:
                    image_tensor = (
                        torch.from_numpy(observation).permute(3, 0, 1, 2).float()
                    )
                    input_tensor = image_tensor.to(self.device) / 255.0
                    with torch.amp.autocast(
                        "cuda", enabled=config.USE_MIXED_PRECISION, dtype=self.dtype
                    ):
                        with torch.no_grad():
                            _, next_value = self.agent(input_tensor.unsqueeze(1))
                            next_value = next_value.squeeze(-1).detach()

                values_tensor = torch.cat(episode_values)
                advantages, returns = calculate_gae(
                    episode_rewards,
                    values_tensor.to(self.device),
                    next_value,
                    config.GAMMA,
                    config.GAE_LAMBDA,
                    self.device,
                )
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
            total_reward = sum(episode_rewards)
            mean_return = returns.mean().item()

            print(
                f"Episode: {self.episode_count}/{config.NUM_EPISODES}, "
                f"Reward: {total_reward:.2f}, "
                f"Avg Return: {mean_return:.2f}, "
                f"Steps: {steps}"
            )
            self.episode_count += 1

        return rollout_buffer

    def ppo_update(self, rollout_buffer):
        self.agent.train()

        actor_losses = []
        critic_losses = []
        for episode in rollout_buffer:
            for i in range(len(episode.states)):
                states = episode.states[i].to(self.device)
                actions = episode.actions[i].to(self.device)
                old_log_probs = episode.log_probs[i].to(self.device)
                advantages = episode.advantages[i].to(self.device)
                returns = episode.returns[i].to(self.device)

                with torch.amp.autocast(
                    "cuda", enabled=config.USE_MIXED_PRECISION, dtype=self.dtype
                ):
                    action_logits, values = self.agent(states.unsqueeze(1))
                    action_dist = Categorical(logits=action_logits)
                    log_probs = action_dist.log_prob(actions)
                    entropy = action_dist.entropy().mean()
                    ratio = torch.exp(log_probs - old_log_probs)
                    surr1 = ratio * advantages
                    surr2 = (
                        torch.clamp(
                            ratio,
                            1.0 - self.config.CLIP_EPSILON,
                            1.0 + self.config.CLIP_EPSILON,
                        )
                        * advantages
                    )
                    actor_loss = -torch.min(surr1, surr2).mean()
                    critic_loss = (
                        F.smooth_l1_loss(values.squeeze(-1), returns)
                        * self.config.VALUE_COEFF
                    )
                    total_loss = (
                        actor_loss + critic_loss - entropy * self.config.ENTROPY_COEFF
                    )

                # Calculate the fraction of clipped samples (This is just for reporting?)
                # with torch.no_grad():
                #     clipped = ratio.gt(1 + self.config.CLIP_EPSILON) | ratio.lt(
                #         1 - self.config.CLIP_EPSILON
                #     )
                #     clip_fraction = torch.as_tensor(clipped, dtype=torch.float32).mean().item()

                self.optimizer.zero_grad()
                if self.scaler is not None:
                    self.scaler.scale(total_loss).backward()
                    self.scaler.unscale_(self.optimizer)
                else:
                    total_loss.backward()

                torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 0.5)

                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                critic_losses.append(critic_loss.item())
                actor_losses.append(actor_loss.item())

        return np.mean(actor_losses), np.mean(critic_losses)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO Training Script")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    args = parser.parse_args()

    config = load_config(args.config)

    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)

    # Extract config name from path (e.g., "baseline" from "configs/baseline.yaml")
    config_name = os.path.splitext(os.path.basename(args.config))[0]

    # Setup MLflow
    mlflow.set_experiment(config_name)
    run_name = f"{config.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run = mlflow.start_run(run_name=run_name)
    mlflow.log_params(config_dict)

    print(f"MLflow experiment '{config_name}' started with run '{run_name}'.")

    manager = PPOManager(config)
    for round in range(1000):
        rollout_buffer = manager.get_rollout()
        actor_loss, critic_loss = manager.ppo_update(rollout_buffer)

        avg_reward = np.mean([np.mean(ep.rewards) for ep in rollout_buffer])
        # mlflow.log_metric("Rewards/Total", total_reward, step=round + 1)
        mlflow.log_metric("Rewards/Average", avg_reward, step=round + 1)
        mlflow.log_metric("Losses/Actor", actor_loss, step=round + 1)
        mlflow.log_metric("Losses/Critic", critic_loss, step=round + 1)
        # mlflow.log_metric("PPO/Entropy", avg_entropy, step=round + 1)
        # mlflow.log_metric("PPO/Clip_Fraction", avg_clip_fraction, step=round + 1)
        # mlflow.log_metric("Value_Function/Mean_Value", mean_value, step=round + 1)
        # mlflow.log_metric("Value_Function/Value_StdDev", value_std, step=round + 1)
        # mlflow.log_metric("Value_Function/Mean_Return", mean_return, step=round + 1)
        # mlflow.log_metric("Performance/Steps_per_Episode", steps, step=round + 1)
