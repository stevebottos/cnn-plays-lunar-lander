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


def ppo_update(
    agent,
    optimizer,
    states,
    actions,
    old_log_probs,
    advantages,
    returns,
    clip_epsilon,
    value_coeff,
    entropy_coeff,
    device,
    use_mixed_precision,
    dtype,
    scaler,
):
    states = states.to(device)
    actions = actions.to(device)
    old_log_probs = old_log_probs.to(device)
    advantages = advantages.to(device)
    returns = returns.to(device)

    with torch.amp.autocast("cuda", enabled=use_mixed_precision, dtype=dtype):
        action_logits, values = agent(states.unsqueeze(1))
        action_dist = Categorical(logits=action_logits)
        log_probs = action_dist.log_prob(actions)
        entropy = action_dist.entropy().mean()
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = F.smooth_l1_loss(values.squeeze(-1), returns) * value_coeff
        total_loss = actor_loss + critic_loss - entropy * entropy_coeff

    # Calculate the fraction of clipped samples
    with torch.no_grad():
        clipped = ratio.gt(1 + clip_epsilon) | ratio.lt(1 - clip_epsilon)
        clip_fraction = torch.as_tensor(clipped, dtype=torch.float32).mean().item()

    optimizer.zero_grad()
    if scaler is not None:
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
    else:
        total_loss.backward()

    torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)

    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()

    return actor_loss.item(), critic_loss.item(), entropy.item(), clip_fraction


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO Training Script")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument(
        "--headless", action="store_true", help="Run in headless mode (no rendering)."
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Custom name for the training run.",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to a checkpoint file to resume training from.",
    )
    parser.add_argument(
        "--resume-run-id",
        type=str,
        default=None,
        help="MLflow run ID to resume.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)

    # Extract config name from path (e.g., "baseline" from "configs/baseline.yaml")
    config_name = os.path.splitext(os.path.basename(args.config))[0]

    # Setup MLflow
    mlflow.set_experiment(config_name)
    if args.resume_run_id:
        run = mlflow.start_run(run_id=args.resume_run_id)
        run_name = run.info.run_name
    else:
        run_name = (
            args.run_name
            or f"{config.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        run = mlflow.start_run(run_name=run_name)
    mlflow.log_params(config_dict)

    print(f"MLflow experiment '{config_name}' started with run '{run_name}'.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config.USE_MIXED_PRECISION and device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
        scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))
    else:
        dtype = torch.float32
        scaler = None

    env = gym.make(
        config.env_name,
        render_mode="rgb_array",
        max_episode_steps=config.MAX_EPISODE_STEPS,
    )
    NUM_ACTIONS = env.action_space.n

    env = AddRenderObservation(env)
    env = ResizeObservation(env, shape=(config.IMAGE_SIZE, config.IMAGE_SIZE))
    env = GrayscaleObservation(env, keep_dim=True)
    env = FrameStackObservation(env, config.NUM_FRAMES)

    if config.model_name == "TinyCNN":
        agent = TinyCNN(num_actions=NUM_ACTIONS).to(device)
    elif config.model_name == "TinyCNNv2":
        agent = TinyCNNv2(num_actions=NUM_ACTIONS).to(device)
    elif config.model_name == "Conv3dResNet":
        agent = Conv3dResNet(num_actions=NUM_ACTIONS).to(device)
    elif config.model_name == "Conv3DTransformerNet":
        agent = Conv3DTransformerNet(
            num_actions=NUM_ACTIONS, num_frames=config.NUM_FRAMES
        ).to(device)
    else:
        raise ValueError(f"Unknown model_name: {config.model_name}")

    if config.USE_TORCH_COMPILE:
        try:
            agent = torch.compile(agent, mode="reduce-overhead")
        except Exception as e:
            pass  # Keep silent if compilation fails

    optimizer = optim.Adam(agent.parameters(), lr=config.LEARNING_RATE)

    start_episode = 0
    if args.resume_run_id:
        client = mlflow.tracking.MlflowClient()
        # Get all artifacts from the 'checkpoints' directory
        try:
            artifacts = client.list_artifacts(args.resume_run_id, "checkpoints")
            if artifacts:
                # Sort artifacts by name (which includes the episode number) to find the latest
                artifacts.sort(key=lambda x: x.path, reverse=True)
                latest_checkpoint_artifact = artifacts[0]

                # Download the latest checkpoint
                checkpoint_path = client.download_artifacts(
                    run_id=args.resume_run_id, path=latest_checkpoint_artifact.path
                )

                checkpoint = torch.load(
                    checkpoint_path, map_location=device, weights_only=False
                )
                agent.load_state_dict(checkpoint["model_state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                start_episode = checkpoint["episode"]
                ppo_round_counter = checkpoint.get("ppo_round_counter", 0)
                print(
                    f"Resumed from checkpoint {latest_checkpoint_artifact.path} in run {args.resume_run_id}"
                )
        except Exception as e:
            print(f"Could not resume from run {args.resume_run_id}: {e}")

    elif args.resume_from:
        if not os.path.isfile(args.resume_from):
            raise FileNotFoundError(f"Checkpoint not found at {args.resume_from}")
        checkpoint = torch.load(
            args.resume_from, map_location=device, weights_only=False
        )
        agent.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_episode = checkpoint["episode"]
        ppo_round_counter = checkpoint.get("ppo_round_counter", 0)
        print(f"Resumed from checkpoint file: {args.resume_from}")

    # This print statement is removed as checkpoints are now handled by MLflow
    # print(f"Checkpoints will be saved to: {run_dir}")

    rollout_states = []
    rollout_actions = []
    rollout_log_probs = []
    rollout_rewards = []
    rollout_advantages = []
    rollout_returns = []
    episode_count = 0
    ppo_round_counter = 0

    for episode in range(start_episode, config.NUM_EPISODES):
        (
            episode_states,
            episode_actions,
            episode_log_probs,
            episode_values,
            episode_rewards,
        ) = (
            [],
            [],
            [],
            [],
            [],
        )
        observation, info = env.reset()
        terminated, truncated = False, False
        agent.eval()
        steps = 0

        while not terminated and not truncated:
            image_tensor = torch.from_numpy(observation).permute(3, 0, 1, 2).float()
            input_tensor = image_tensor.to(device) / 255.0
            episode_states.append(input_tensor.cpu())
            with torch.amp.autocast(
                "cuda", enabled=config.USE_MIXED_PRECISION, dtype=dtype
            ):
                with torch.no_grad():
                    action_logits, value = agent(input_tensor.unsqueeze(0))
            action_dist = Categorical(logits=action_logits.float())
            action = action_dist.sample()
            episode_actions.append(action.detach().cpu())
            episode_log_probs.append(action_dist.log_prob(action).detach().cpu())
            episode_values.append(value.squeeze(-1).detach().cpu())
            observation, reward, terminated, truncated, info = env.step(action.item())
            clipped_reward = np.clip(reward, -config.REWARD_CLIP, config.REWARD_CLIP)
            episode_rewards.append(clipped_reward)
            steps += 1
            if not args.headless:
                display_frame = observation[-1, :, :, :]
                display_frame = np.interp(
                    display_frame, (display_frame.min(), display_frame.max()), (0, 255)
                ).astype(np.uint8)
                cv2.imshow("Agent View (84x84 Grayscale)", display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    sys.exit("Simulation stopped by user.")

        if episode_rewards:
            if terminated:
                next_value = torch.tensor(0.0, device=device)
            else:
                image_tensor = torch.from_numpy(observation).permute(3, 0, 1, 2).float()
                input_tensor = image_tensor.to(device) / 255.0
                with torch.amp.autocast(
                    "cuda", enabled=config.USE_MIXED_PRECISION, dtype=dtype
                ):
                    with torch.no_grad():
                        _, next_value = agent(input_tensor.unsqueeze(1))
                        next_value = next_value.squeeze(-1).detach()
            values_tensor = torch.cat(episode_values)
            advantages, returns = calculate_gae(
                episode_rewards,
                values_tensor.to(device),
                next_value,
                config.GAMMA,
                config.GAE_LAMBDA,
                device,
            )
            rollout_states.extend(episode_states)
            rollout_actions.extend(episode_actions)
            rollout_log_probs.extend(episode_log_probs)
            rollout_rewards.append(sum(episode_rewards))
            rollout_advantages.append(advantages.cpu())
            rollout_returns.append(returns.cpu())
            episode_count += 1
            total_reward = sum(episode_rewards)
            mean_value = values_tensor.mean().item()
            value_std = values_tensor.std().item()
            mean_return = returns.mean().item()

            print(
                f"Episode: {episode + 1}/{config.NUM_EPISODES}, "
                f"Reward: {total_reward:.2f}, "
                f"Avg Return: {mean_return:.2f}, "
                f"Steps: {steps}"
            )

        if episode_count >= config.COLLECT_EPISODES:
            agent.train()
            states_batch = torch.cat(rollout_states)
            actions_batch = torch.cat(rollout_actions)
            old_log_probs_batch = torch.cat(rollout_log_probs)
            advantages_batch = torch.cat(rollout_advantages)
            returns_batch = torch.cat(rollout_returns)
            advantages_batch = (advantages_batch - advantages_batch.mean()) / (
                advantages_batch.std() + 1e-8
            )
            dataset_size = len(states_batch)
            indices = np.arange(dataset_size)
            (
                total_actor_loss,
                total_critic_loss,
                total_entropy,
                num_updates,
                total_clip_fraction,
            ) = (
                0,
                0,
                0,
                0,
                0,
            )

            for epoch in range(config.PPO_EPOCHS):
                np.random.shuffle(indices)
                for start in range(0, dataset_size, config.BATCH_SIZE):
                    end = min(start + config.BATCH_SIZE, dataset_size)
                    batch_indices = indices[start:end]
                    states_mb = states_batch[batch_indices]
                    actions_mb = actions_batch[batch_indices]
                    old_log_probs_mb = old_log_probs_batch[batch_indices]
                    advantages_mb = advantages_batch[batch_indices]
                    returns_mb = returns_batch[batch_indices]
                    actor_loss, critic_loss, entropy, clip_fraction = ppo_update(
                        agent,
                        optimizer,
                        states_mb,
                        actions_mb,
                        old_log_probs_mb,
                        advantages_mb,
                        returns_mb,
                        config.CLIP_EPSILON,
                        config.VALUE_COEFF,
                        config.ENTROPY_COEFF,
                        device,
                        config.USE_MIXED_PRECISION,
                        dtype,
                        scaler,
                    )
                    total_actor_loss += actor_loss
                    total_critic_loss += critic_loss
                    total_entropy += entropy
                    total_clip_fraction += clip_fraction
                    num_updates += 1

            avg_actor_loss = total_actor_loss / num_updates
            avg_critic_loss = total_critic_loss / num_updates
            avg_entropy = total_entropy / num_updates
            avg_clip_fraction = total_clip_fraction / num_updates
            avg_reward = np.mean(rollout_rewards)
            ppo_round_counter += 1

            # Log metrics to MLflow
            mlflow.log_metric("Rewards/Total", total_reward, step=episode + 1)
            mlflow.log_metric("Rewards/Average", avg_reward, step=episode + 1)
            mlflow.log_metric("Losses/Actor", avg_actor_loss, step=episode + 1)
            mlflow.log_metric("Losses/Critic", avg_critic_loss, step=episode + 1)
            mlflow.log_metric("PPO/Entropy", avg_entropy, step=episode + 1)
            mlflow.log_metric("PPO/Clip_Fraction", avg_clip_fraction, step=episode + 1)
            mlflow.log_metric("Value_Function/Mean_Value", mean_value, step=episode + 1)
            mlflow.log_metric(
                "Value_Function/Value_StdDev", value_std, step=episode + 1
            )
            mlflow.log_metric(
                "Value_Function/Mean_Return", mean_return, step=episode + 1
            )
            mlflow.log_metric("Performance/Steps_per_Episode", steps, step=episode + 1)

            # Clear rollout buffers
            (
                rollout_states,
                rollout_actions,
                rollout_log_probs,
                rollout_rewards,
                rollout_advantages,
                rollout_returns,
            ) = (
                [],
                [],
                [],
                [],
                [],
                [],
            )
            episode_count = 0

            if ppo_round_counter % config.CHECKPOINT_PPO_ROUNDS == 0:
                checkpoint = {
                    "episode": episode + 1,
                    "ppo_round_counter": ppo_round_counter,
                    "model_state_dict": agent.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "avg_reward": avg_reward,
                }
                with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
                    torch.save(checkpoint, tmp.name)
                    mlflow.log_artifact(
                        tmp.name,
                        artifact_path=f"checkpoints/checkpoint_ppo_round_{ppo_round_counter:07d}.pt",
                    )
                os.remove(tmp.name)  # Clean up the temporary file
                print(
                    f"Saved checkpoint for PPO round {ppo_round_counter} to MLflow artifacts."
                )

        observation, info = env.reset()

    env.close()
    if not args.headless:
        cv2.destroyAllWindows()

    checkpoint = {
        "episode": episode + 1,
        "model_state_dict": agent.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        torch.save(checkpoint, tmp.name)
        mlflow.log_artifact(
            tmp.name,
            artifact_path=f"checkpoints/checkpoint_{episode + 1:07d}.pt",
        )
    os.remove(tmp.name)  # Clean up the temporary file
    print(f"Saved final checkpoint for episode {episode + 1} to MLflow artifacts.")
    mlflow.end_run()
