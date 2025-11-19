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
import numpy as np
import cv2
import sys
import argparse
import yaml
import glob
import os

from models import Conv3DTransformerNet, Conv3dResNet, TinyCNN

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

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
    scaler
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

    return actor_loss.item(), critic_loss.item(), entropy.item()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PPO Training Script")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    args = parser.parse_args()

    config = load_config(args.config)
    
    USE_MIXED_PRECISION = config['USE_MIXED_PRECISION']
    USE_TORCH_COMPILE = config['USE_TORCH_COMPILE']
    HEADLESS = config['HEADLESS']
    CHECKPOINT_INTERVAL = config['CHECKPOINT_INTERVAL']
    LEARNING_RATE = config['LEARNING_RATE']
    GAMMA = config['GAMMA']
    GAE_LAMBDA = config['GAE_LAMBDA']
    NUM_EPISODES = config['NUM_EPISODES']
    VALUE_COEFF = config['VALUE_COEFF']
    ENTROPY_COEFF = config['ENTROPY_COEFF']
    REWARD_CLIP = config['REWARD_CLIP']
    CLIP_EPSILON = config['CLIP_EPSILON']
    BATCH_SIZE = config['BATCH_SIZE']
    PPO_EPOCHS = config['PPO_EPOCHS']
    COLLECT_EPISODES = config['COLLECT_EPISODES']
    LOG_FILE = config['LOG_FILE']
    IMAGE_SIZE = config['IMAGE_SIZE']
    NUM_FRAMES = config['NUM_FRAMES']
    MAX_EPISODE_STEPS = config['MAX_EPISODE_STEPS']
    env_name = config['env_name']
    model_name = config['model_name']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if USE_MIXED_PRECISION and device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
            print("Using bfloat16 mixed precision training")
        else:
            dtype = torch.float16
            print("Using float16 mixed precision training")
        scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))
    else:
        dtype = torch.float32
        scaler = None
        print("Using float32 (no mixed precision)")

    env = gym.make(env_name, render_mode="rgb_array", max_episode_steps=MAX_EPISODE_STEPS)
    NUM_ACTIONS = env.action_space.n

    env = AddRenderObservation(env)
    env = ResizeObservation(env, shape=(IMAGE_SIZE, IMAGE_SIZE))
    env = GrayscaleObservation(env, keep_dim=True)
    env = FrameStackObservation(env, NUM_FRAMES)

    print(f"Using {model_name} architecture")
    if model_name == "TinyCNN":
        agent = TinyCNN(num_actions=NUM_ACTIONS).to(device)
    elif model_name == "Conv3dResNet":
        agent = Conv3dResNet(num_actions=NUM_ACTIONS).to(device)
    elif model_name == "Conv3DTransformerNet":
        agent = Conv3DTransformerNet(num_actions=NUM_ACTIONS, num_frames=NUM_FRAMES).to(device)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    if USE_TORCH_COMPILE:
        try:
            agent = torch.compile(agent, mode="reduce-overhead")
            print("Model compiled with torch.compile()")
        except Exception as e:
            print(f"torch.compile() failed: {e}, continuing without compilation")

    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE)

    start_episode = 0
    checkpoint_files = sorted(glob.glob("checkpoints/checkpoint_*.pt"))
    if checkpoint_files:
        latest_checkpoint = checkpoint_files[-1]
        print(f"Loading checkpoint: {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint, map_location=device, weights_only=False)
        agent.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_episode = checkpoint["episode"]
        print(f"Resuming from episode {start_episode}")

        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "r") as f:
                lines = f.readlines()
            header = lines[0]
            data_lines = lines[1:]
            truncated_lines = [header]
            for line in data_lines:
                episode_num = int(line.split(",")[0])
                if episode_num <= start_episode:
                    truncated_lines.append(line)
            with open(LOG_FILE, "w") as f:
                f.writelines(truncated_lines)
            print(f"Truncated log file to episode {start_episode}")
    else:
        print("No checkpoint found, starting from scratch")

    print("\nStarting Proximal Policy Optimization (PPO) simulation loop...")

    if start_episode > 0:
        log_file = open(LOG_FILE, "a")
        print(f"Appending to existing log file: {LOG_FILE}")
    else:
        log_file = open(LOG_FILE, "w")
        log_file.write(
            "Episode,Steps,TotalReward,ActorLoss,CriticLoss,MeanValue,ValueStd,MeanReturn,ClipFraction\n"
        )
        print(f"Created new log file: {LOG_FILE}")

    rollout_states = []
    rollout_actions = []
    rollout_log_probs = []
    rollout_rewards = []
    rollout_advantages = []
    rollout_returns = []
    episode_count = 0

    for episode in range(start_episode, NUM_EPISODES):
        episode_states, episode_actions, episode_log_probs, episode_values, episode_rewards = [], [], [], [], []
        observation, info = env.reset()
        terminated, truncated = False, False
        agent.eval()
        steps = 0

        while not terminated and not truncated:
            image_tensor = torch.from_numpy(observation).permute(3, 0, 1, 2).float()
            input_tensor = image_tensor.to(device) / 255.0
            episode_states.append(input_tensor.cpu())
            with torch.amp.autocast("cuda", enabled=USE_MIXED_PRECISION, dtype=dtype):
                with torch.no_grad():
                    action_logits, value = agent(input_tensor.unsqueeze(0))
            action_dist = Categorical(logits=action_logits.float())
            action = action_dist.sample()
            episode_actions.append(action.detach().cpu())
            episode_log_probs.append(action_dist.log_prob(action).detach().cpu())
            episode_values.append(value.squeeze(-1).detach().cpu())
            observation, reward, terminated, truncated, info = env.step(action.item())
            clipped_reward = np.clip(reward, -REWARD_CLIP, REWARD_CLIP)
            episode_rewards.append(clipped_reward)
            steps += 1
            if not HEADLESS:
                display_frame = observation[-1, :, :, :]
                display_frame = np.interp(display_frame, (display_frame.min(), display_frame.max()), (0, 255)).astype(np.uint8)
                cv2.imshow("Agent View (84x84 Grayscale)", display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    sys.exit("Simulation stopped by user.")

        if episode_rewards:
            if terminated:
                next_value = torch.tensor(0.0, device=device)
            else:
                image_tensor = torch.from_numpy(observation).permute(3, 0, 1, 2).float()
                input_tensor = image_tensor.to(device) / 255.0
                with torch.amp.autocast("cuda", enabled=USE_MIXED_PRECISION, dtype=dtype):
                    with torch.no_grad():
                        _, next_value = agent(input_tensor.unsqueeze(1))
                        next_value = next_value.squeeze(-1).detach()
            values_tensor = torch.cat(episode_values)
            advantages, returns = calculate_gae(episode_rewards, values_tensor.to(device), next_value, GAMMA, GAE_LAMBDA, device)
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
                f"Episode {episode + 1:4d} | Steps: {steps:3d} | Total Reward: {total_reward:6.2f} | "
                f"Value: {mean_value:6.2f}±{value_std:.2f} | Return: {mean_return:6.2f}"
            )

        if episode_count >= COLLECT_EPISODES:
            agent.train()
            print(f"  Collected {len(rollout_states)} states, {len(rollout_advantages)} advantages")
            states_batch = torch.cat(rollout_states)
            actions_batch = torch.cat(rollout_actions)
            old_log_probs_batch = torch.cat(rollout_log_probs)
            advantages_batch = torch.cat(rollout_advantages)
            returns_batch = torch.cat(rollout_returns)
            advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)
            dataset_size = len(states_batch)
            indices = np.arange(dataset_size)
            total_actor_loss, total_critic_loss, total_entropy, num_updates = 0, 0, 0, 0

            for epoch in range(PPO_EPOCHS):
                np.random.shuffle(indices)
                for start in range(0, dataset_size, BATCH_SIZE):
                    end = min(start + BATCH_SIZE, dataset_size)
                    batch_indices = indices[start:end]
                    states_mb = states_batch[batch_indices]
                    actions_mb = actions_batch[batch_indices]
                    old_log_probs_mb = old_log_probs_batch[batch_indices]
                    advantages_mb = advantages_batch[batch_indices]
                    returns_mb = returns_batch[batch_indices]
                    actor_loss, critic_loss, entropy = ppo_update(
                        agent, optimizer, states_mb, actions_mb, old_log_probs_mb, advantages_mb, returns_mb,
                        CLIP_EPSILON, VALUE_COEFF, ENTROPY_COEFF, device, USE_MIXED_PRECISION, dtype, scaler
                    )
                    total_actor_loss += actor_loss
                    total_critic_loss += critic_loss
                    total_entropy += entropy
                    num_updates += 1

            clip_fraction = 0.0
            avg_actor_loss = total_actor_loss / num_updates
            avg_critic_loss = total_critic_loss / num_updates
            avg_entropy = total_entropy / num_updates
            avg_reward = np.mean(rollout_rewards)
            print(
                f"  PPO Update | Avg Reward: {avg_reward:6.2f} | "
                f"Actor Loss: {avg_actor_loss:.4f} | Critic Loss: {avg_critic_loss:.4f} | "
                f"Entropy: {avg_entropy:.4f} | ClipFrac: {clip_fraction:.2f}"
            )
            log_file.write(
                f"{episode + 1},{steps},{avg_reward:.2f},{avg_actor_loss:.4f},"
                f"{avg_critic_loss:.4f},{mean_value:.2f},{value_std:.2f},"
                f"{mean_return:.2f},{clip_fraction:.4f}\n"
            )
            log_file.flush()
            rollout_states, rollout_actions, rollout_log_probs, rollout_rewards, rollout_advantages, rollout_returns = [], [], [], [], [], []
            episode_count = 0

            if (episode + 1) % CHECKPOINT_INTERVAL == 0:
                os.makedirs("checkpoints", exist_ok=True)
                checkpoint = {
                    "episode": episode + 1,
                    "model_state_dict": agent.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "avg_reward": avg_reward,
                }
                checkpoint_path = f"checkpoints/checkpoint_{episode + 1:07d}.pt"
                torch.save(checkpoint, checkpoint_path)
                print(f"  Checkpoint saved: {checkpoint_path}")

        observation, info = env.reset()

    log_file.close()
    env.close()
    if not HEADLESS:
        cv2.destroyAllWindows()

    os.makedirs("checkpoints", exist_ok=True)
    checkpoint = {
        "episode": episode + 1,
        "model_state_dict": agent.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    checkpoint_path = f"checkpoints/checkpoint_{episode + 1:07d}.pt"
    torch.save(checkpoint, checkpoint_path)
    print(f"Final checkpoint saved: {checkpoint_path}")
    print(f"\nSimulation complete. Environment closed. Logs saved to {LOG_FILE}")
