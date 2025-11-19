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

from models import Conv3DTransformerNet, Conv3dResNet, TinyCNN

# --- Configuration and Hyperparameters ---
USE_MIXED_PRECISION = True  # Use bfloat16 automatic mixed precision
USE_TORCH_COMPILE = False  # torch.compile() for speedup (requires PyTorch 2.0+)
HEADLESS = True  # Set to True to disable cv2 visualization window
CHECKPOINT_INTERVAL = 1000  # Save checkpoint every N episodes

# PPO-specific hyperparameters
LEARNING_RATE = 3e-4  # PPO can handle higher learning rates due to clipping
GAMMA = 0.99
GAE_LAMBDA = 0.95  # Lambda for Generalized Advantage Estimation
NUM_EPISODES = 100000
VALUE_COEFF = 0.01  # Coefficient for the value function loss
ENTROPY_COEFF = 0.01  # Coefficient for the entropy term
REWARD_CLIP = 5.0  # Clip rewards to [-REWARD_CLIP, +REWARD_CLIP] for stability

# PPO-specific parameters
CLIP_EPSILON = 0.2  # PPO clipping parameter
BATCH_SIZE = 64  # Minibatch size for PPO updates
PPO_EPOCHS = 4  # Number of epochs to train on collected data
COLLECT_EPISODES = 16  # Number of episodes to collect before updating

LOG_FILE = "training_log_ppo.txt"  # Log file for training metrics
IMAGE_SIZE = 128
NUM_FRAMES = 16
MAX_EPISODE_STEPS = 500
# Setup device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Mixed precision setup
if USE_MIXED_PRECISION and device.type == "cuda":
    # Use bfloat16 if available (better than float16 for training)
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

env_name = "LunarLander-v3"
env = gym.make(env_name, render_mode="rgb_array", max_episode_steps=MAX_EPISODE_STEPS)
NUM_ACTIONS = env.action_space.n  # pyright: ignore

# Apply wrappers for visual processing (as in your original code)
env = AddRenderObservation(env)
env = ResizeObservation(env, shape=(IMAGE_SIZE, IMAGE_SIZE))
env = GrayscaleObservation(env, keep_dim=True)
env = FrameStackObservation(env, NUM_FRAMES)
# --- 3. Model Initialization ---

# Initialize model
print("Using Conv3dResNet architecture")
agent = TinyCNN(num_actions=NUM_ACTIONS).to(device)

# Optional: Compile model for speedup (PyTorch 2.0+)
if USE_TORCH_COMPILE:
    try:
        agent = torch.compile(agent, mode="reduce-overhead")
        print("Model compiled with torch.compile()")
    except Exception as e:
        print(f"torch.compile() failed: {e}, continuing without compilation")

# Optimizer
optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE)

# --- Load checkpoint if available ---
import glob
import os

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

    # Truncate training log to remove entries after start_episode
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            lines = f.readlines()

        # Keep header + lines up to start_episode
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

# --- 4. Helper Functions for PPO ---


def calculate_gae(rewards, values, next_value, gamma, gae_lambda):
    """
    Calculate Generalized Advantage Estimation (GAE).

    GAE provides a better bias-variance tradeoff than pure Monte Carlo returns.
    It's a weighted average of n-step advantages.

    Args:
        rewards: List of rewards for the episode
        values: Tensor of value predictions V(s_t)
        next_value: Value of the next state (0 if terminal)
        gamma: Discount factor
        gae_lambda: GAE lambda parameter (higher = more Monte Carlo, lower = more TD)

    Returns:
        advantages: Tensor of advantage estimates
        returns: Tensor of return targets for value function
    """
    advantages = []
    gae = 0

    # Convert rewards to tensor
    rewards = torch.tensor(rewards, dtype=torch.float32, device=device)

    # Ensure next_value is a 0-d tensor
    if isinstance(next_value, (int, float)):
        next_value = torch.tensor(next_value, dtype=torch.float32, device=device)
    elif next_value.dim() > 0:
        next_value = next_value.squeeze()

    # Append next_value for bootstrapping
    values_list = torch.cat([values, next_value.unsqueeze(0)])

    # Calculate GAE backwards through episode
    for t in reversed(range(len(rewards))):
        # TD error: delta = r + gamma * V(s_{t+1}) - V(s_t)
        delta = rewards[t] + gamma * values_list[t + 1] - values_list[t]
        # GAE: A_t = delta + gamma * lambda * A_{t+1}
        gae = delta + gamma * gae_lambda * gae
        advantages.insert(0, gae)

    advantages = torch.stack(advantages)

    # Returns are advantages + values (for value function target)
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
):
    """
    Perform PPO update using clipped objective.

    Args:
        agent: The policy network
        optimizer: The optimizer
        states: Batch of states
        actions: Batch of actions taken
        old_log_probs: Log probs of actions under old policy
        advantages: Advantage estimates
        returns: Return targets for value function
        clip_epsilon: PPO clipping parameter
        value_coeff: Coefficient for value loss
        entropy_coeff: Coefficient for entropy bonus

    Returns:
        actor_loss, critic_loss, entropy: Loss values for logging
    """
    # Move mini-batch to device
    states = states.to(device)
    actions = actions.to(device)
    old_log_probs = old_log_probs.to(device)
    advantages = advantages.to(device)
    returns = returns.to(device)

    # Forward pass
    with torch.amp.autocast("cuda", enabled=USE_MIXED_PRECISION, dtype=dtype):
        action_logits, values = agent(states.unsqueeze(1))
        # Get current policy distribution
        action_dist = Categorical(logits=action_logits)
        log_probs = action_dist.log_prob(actions)
        entropy = action_dist.entropy().mean()

        # PPO clipped objective
        # ratio = π_new(a|s) / π_old(a|s)
        ratio = torch.exp(log_probs - old_log_probs)

        # Clipped surrogate objective
        # L^CLIP = min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()

        # Value function loss (Huber loss for robustness)
        critic_loss = F.smooth_l1_loss(values.squeeze(-1), returns) * value_coeff

        # Total loss
        total_loss = actor_loss + critic_loss - entropy * entropy_coeff

    # Backward pass
    optimizer.zero_grad()
    if scaler is not None:
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
    else:
        total_loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)

    # Optimizer step
    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()

    return actor_loss.item(), critic_loss.item(), entropy.item()


# --- 5. Training Loop ---

print("\nStarting Proximal Policy Optimization (PPO) simulation loop...")

# Open log file (append if resuming, write if starting fresh)
if start_episode > 0:
    log_file = open(LOG_FILE, "a")
    print(f"Appending to existing log file: {LOG_FILE}")
else:
    log_file = open(LOG_FILE, "w")
    log_file.write(
        "Episode,Steps,TotalReward,ActorLoss,CriticLoss,MeanValue,ValueStd,MeanReturn,ClipFraction\n"
    )
    print(f"Created new log file: {LOG_FILE}")

# Rollout buffer for collecting multiple episodes
rollout_states = []
rollout_actions = []
rollout_log_probs = []
rollout_rewards = []
rollout_advantages = []
rollout_returns = []

episode_count = 0

for episode in range(start_episode, NUM_EPISODES):
    # Data storage for the episode
    episode_states = []
    episode_actions = []
    episode_log_probs = []
    episode_values = []
    episode_rewards = []

    # Reset environment
    observation, info = env.reset()
    terminated = False
    truncated = False

    # Set model to eval mode for episode collection
    agent.eval()

    steps = 0
    while not terminated and not truncated:
        image_tensor = torch.from_numpy(observation).permute(3, 0, 1, 2).float()
        input_tensor = image_tensor.to(device) / 255.0

        # Store state on CPU to save GPU memory
        episode_states.append(input_tensor.cpu())

        # Forward pass with mixed precision
        with torch.amp.autocast("cuda", enabled=USE_MIXED_PRECISION, dtype=dtype):
            with torch.no_grad():  # No gradients during rollout
                action_logits, value = agent(input_tensor.unsqueeze(0))

        # Sample action from policy
        action_dist = Categorical(logits=action_logits.float())
        action = action_dist.sample()

        # Store data on CPU
        episode_actions.append(action.detach().cpu())
        episode_log_probs.append(action_dist.log_prob(action).detach().cpu())
        episode_values.append(value.squeeze(-1).detach().cpu())

        # Step environment
        observation, reward, terminated, truncated, info = env.step(action.item())

        # Clip reward for stability
        clipped_reward = np.clip(reward, -REWARD_CLIP, REWARD_CLIP)
        episode_rewards.append(clipped_reward)

        steps += 1

        # Render the agent's grayscale view (if not headless)
        if not HEADLESS:
            display_frame = observation[-1, :, :, :]
            display_frame = np.interp(
                display_frame, (display_frame.min(), display_frame.max()), (0, 255)
            ).astype(np.uint8)
            cv2.imshow("Agent View (84x84 Grayscale)", display_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                sys.exit("Simulation stopped by user.")

    # Episode finished - add to rollout buffer
    if episode_rewards:  # Only if episode had steps
        # Calculate next value for bootstrapping (0 if terminal)
        if terminated:
            next_value = torch.tensor(0.0, device=device)
        else:
            # If truncated, bootstrap from last state
            image_tensor = torch.from_numpy(observation).permute(3, 0, 1, 2).float()
            input_tensor = image_tensor.to(device) / 255.0
            with torch.amp.autocast("cuda", enabled=USE_MIXED_PRECISION, dtype=dtype):
                with torch.no_grad():
                    _, next_value = agent(input_tensor.unsqueeze(1))
                    next_value = next_value.squeeze(-1).detach()

        # Calculate GAE advantages and returns
        values_tensor = torch.cat(episode_values)
        advantages, returns = calculate_gae(
            episode_rewards, values_tensor.to(device), next_value, GAMMA, GAE_LAMBDA
        )

        # Add to rollout buffer
        rollout_states.extend(episode_states)
        rollout_actions.extend(episode_actions)
        rollout_log_probs.extend(episode_log_probs)

        rollout_rewards.append(sum(episode_rewards))

        # Store advantages and returns (per timestep)
        rollout_advantages.append(advantages.cpu())
        rollout_returns.append(returns.cpu())

        episode_count += 1

        # Log episode info
        total_reward = sum(episode_rewards)
        mean_value = values_tensor.mean().item()
        value_std = values_tensor.std().item()
        mean_return = returns.mean().item()

        print(
            f"Episode {episode + 1:4d} | Steps: {steps:3d} | Total Reward: {total_reward:6.2f} | "
            f"Value: {mean_value:6.2f}±{value_std:.2f} | Return: {mean_return:6.2f}"
        )

    # Perform PPO update after collecting enough episodes
    if episode_count >= COLLECT_EPISODES:
        # Set model to train mode for PPO updates
        agent.train()

        # Debug: Check buffer sizes
        print(
            f"  Collected {len(rollout_states)} states, {len(rollout_advantages)} advantages"
        )

        # Convert rollout buffer to tensors
        states_batch = torch.cat(rollout_states)
        actions_batch = torch.cat(rollout_actions)
        old_log_probs_batch = torch.cat(rollout_log_probs)
        advantages_batch = torch.cat(rollout_advantages)
        returns_batch = torch.cat(rollout_returns)

        # Normalize advantages
        advantages_batch = (advantages_batch - advantages_batch.mean()) / (
            advantages_batch.std() + 1e-8
        )

        # Perform multiple epochs of optimization
        dataset_size = len(states_batch)
        indices = np.arange(dataset_size)

        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        num_updates = 0

        for epoch in range(PPO_EPOCHS):
            np.random.shuffle(indices)

            # Mini-batch updates
            for start in range(0, dataset_size, BATCH_SIZE):
                end = min(start + BATCH_SIZE, dataset_size)
                batch_indices = indices[start:end]

                # Get mini-batch
                states_mb = states_batch[batch_indices]
                actions_mb = actions_batch[batch_indices]
                old_log_probs_mb = old_log_probs_batch[batch_indices]
                advantages_mb = advantages_batch[batch_indices]
                returns_mb = returns_batch[batch_indices]

                # PPO update
                actor_loss, critic_loss, entropy = ppo_update(
                    agent,
                    optimizer,
                    states_mb,
                    actions_mb,
                    old_log_probs_mb,
                    advantages_mb,
                    returns_mb,
                    CLIP_EPSILON,
                    VALUE_COEFF,
                    ENTROPY_COEFF,
                )

                total_actor_loss += actor_loss
                total_critic_loss += critic_loss
                total_entropy += entropy
                num_updates += 1

        # Clip fraction removed to avoid OOM with large batches
        clip_fraction = 0.0

        # Log update info
        avg_actor_loss = total_actor_loss / num_updates
        avg_critic_loss = total_critic_loss / num_updates
        avg_entropy = total_entropy / num_updates
        avg_reward = np.mean(rollout_rewards)

        print(
            f"  PPO Update | Avg Reward: {avg_reward:6.2f} | "
            f"Actor Loss: {avg_actor_loss:.4f} | Critic Loss: {avg_critic_loss:.4f} | "
            f"Entropy: {avg_entropy:.4f} | ClipFrac: {clip_fraction:.2f}"
        )

        # Write to log file (one entry per update)
        log_file.write(
            f"{episode + 1},{steps},{avg_reward:.2f},{avg_actor_loss:.4f},"
            f"{avg_critic_loss:.4f},{mean_value:.2f},{value_std:.2f},"
            f"{mean_return:.2f},{clip_fraction:.4f}\n"
        )
        log_file.flush()

        # Clear rollout buffer
        rollout_states = []
        rollout_actions = []
        rollout_log_probs = []
        rollout_rewards = []
        rollout_advantages = []
        rollout_returns = []
        episode_count = 0

        # Save checkpoint periodically
        if (episode + 1) % CHECKPOINT_INTERVAL == 0:
            import os

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

    # Reset environment for next episode
    observation, info = env.reset()

# --- Cleanup ---
log_file.close()
env.close()
if not HEADLESS:
    cv2.destroyAllWindows()

# Save final checkpoint
import os

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
