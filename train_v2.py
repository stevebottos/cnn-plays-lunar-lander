# type: ignore
from pathlib import Path
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
import numpy as np
import argparse
import yaml
import os
from datetime import datetime
from models import Conv3DTransformerNet, Conv3dResNet, TinyCNN, TinyCNNv2
from configs.config import TrainingConfig
from collections import namedtuple


def load_config(config_path: str) -> TrainingConfig:
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    return TrainingConfig(**config_dict), config_dict


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
                            _, next_value = self.agent(input_tensor.unsqueeze(0))
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
                f"Episode: {self.episode_count}, "
                f"Reward: {total_reward:.2f}, "
                f"Avg Return: {mean_return:.2f}, "
                f"Steps: {steps}"
            )
            self.episode_count += 1

        return rollout_buffer

    def ppo_update(self, rollout_buffer):
        self.agent.train()

        # 1. Aggregate data from rollout buffer
        states_batch = torch.stack(
            [s for ep in rollout_buffer for s in ep.states]
        ).cpu()
        actions_batch = torch.cat(
            [torch.tensor(ep.actions) for ep in rollout_buffer]
        ).cpu()
        old_log_probs_batch = torch.stack(
            [lp for ep in rollout_buffer for lp in ep.log_probs]
        ).cpu()
        advantages_batch = torch.cat([ep.advantages for ep in rollout_buffer]).cpu()
        returns_batch = torch.cat([ep.returns for ep in rollout_buffer]).cpu()

        # 2. Normalize advantages
        advantages_batch = (advantages_batch - advantages_batch.mean()) / (
            advantages_batch.std() + 1e-8
        )

        total_actor_loss = 0
        total_critic_loss = 0
        num_updates = 0
        dataset_size = len(states_batch)
        indices = np.arange(dataset_size)

        # 3. PPO Epochs loop
        for epoch in range(self.config.PPO_EPOCHS):
            # 4. Shuffle indices
            np.random.shuffle(indices)

            # 5. Mini-batch loop
            for start in range(0, dataset_size, self.config.BATCH_SIZE):
                end = start + self.config.BATCH_SIZE
                batch_indices = indices[start:end]

                # 6. Get mini-batch data
                states_mb = states_batch[batch_indices].to(self.device)
                actions_mb = actions_batch[batch_indices].to(self.device)
                old_log_probs_mb = old_log_probs_batch[batch_indices].to(self.device)
                advantages_mb = advantages_batch[batch_indices].to(self.device)
                returns_mb = returns_batch[batch_indices].to(self.device)

                with torch.amp.autocast(
                    "cuda", enabled=self.config.USE_MIXED_PRECISION, dtype=self.dtype
                ):
                    action_logits, values = self.agent(states_mb)
                    action_dist = Categorical(logits=action_logits)
                    log_probs = action_dist.log_prob(actions_mb)
                    entropy = action_dist.entropy().mean()
                    ratio = torch.exp(log_probs - old_log_probs_mb)
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
                    critic_loss = (
                        F.smooth_l1_loss(values.squeeze(-1), returns_mb)
                        * self.config.VALUE_COEFF
                    )
                    total_loss = (
                        actor_loss + critic_loss - entropy * self.config.ENTROPY_COEFF
                    )

                self.optimizer.zero_grad()
                if self.scaler:
                    self.scaler.scale(total_loss).backward()
                    self.scaler.unscale_(self.optimizer)
                else:
                    total_loss.backward()

                torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 0.5)

                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                num_updates += 1

        return total_actor_loss / num_updates, total_critic_loss / num_updates


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO Training Script")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    args = parser.parse_args()
    config, config_as_dict = load_config(args.config)
    config_name = os.path.splitext(os.path.basename(args.config))[0]

    checkpoints_out = Path("checkpoints")
    checkpoints_out.mkdir(parents=True, exist_ok=True)

    # Setup MLflow
    mlflow.set_experiment(config_name)
    run_name = f"{config.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run = mlflow.start_run(run_name=run_name)
    mlflow.log_params(config_as_dict)

    print(f"MLflow experiment '{config_name}' started with run '{run_name}'.")

    manager = PPOManager(config)
    for round in range(config.NUM_ROUNDS):
        rollout_buffer = manager.get_rollout()
        actor_loss, critic_loss = manager.ppo_update(rollout_buffer)
        avg_reward = np.mean([np.sum(ep.rewards) for ep in rollout_buffer])
        mlflow.log_metric("Rewards/Average", avg_reward, step=round + 1)
        mlflow.log_metric("Losses/Actor", actor_loss, step=round + 1)
        mlflow.log_metric("Losses/Critic", critic_loss, step=round + 1)

        if round % 250 == 0:
            checkpoint_path = checkpoints_out / f"{str(round).zfill(7)}.pt"
            torch.save(manager.agent.state_dict(), str(checkpoint_path))
