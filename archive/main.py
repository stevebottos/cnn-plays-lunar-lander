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

from models import Conv3DTransformerNet

# --- Configuration and Hyperparameters ---
USE_MIXED_PRECISION = True  # Use bfloat16 automatic mixed precision
USE_TORCH_COMPILE = False  # torch.compile() for speedup (requires PyTorch 2.0+)

LEARNING_RATE = 1e-5  # Single learning rate
GAMMA = 0.99
NUM_EPISODES = 10000
VALUE_COEFF = 0.5  # Coefficient for the value function loss
ENTROPY_COEFF = 0.01  # Coefficient for the entropy term
REWARD_CLIP = 3.0  # Clip rewards to [-REWARD_CLIP, +REWARD_CLIP] for stability
LOG_FILE = "training_log_conv3d_transformer.txt"  # Log file for training metrics

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
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == torch.float16))
else:
    dtype = torch.float32
    scaler = None
    print("Using float32 (no mixed precision)")

env_name = "LunarLander-v3"
env = gym.make(env_name, render_mode="rgb_array")
NUM_ACTIONS = env.action_space.n  # pyright: ignore

# Apply wrappers for visual processing (as in your original code)
env = AddRenderObservation(env)
env = ResizeObservation(env, shape=(84, 84))
env = GrayscaleObservation(env, keep_dim=True)  # Output is (H, W, 1)
env = FrameStackObservation(env, 8)
# --- 3. Model Initialization ---

# Initialize model
print("Using Conv3D + Transformer architecture")
agent = Conv3DTransformerNet(num_actions=NUM_ACTIONS).to(device)

# Optional: Compile model for speedup (PyTorch 2.0+)
if USE_TORCH_COMPILE:
    try:
        agent = torch.compile(agent, mode="reduce-overhead")
        print("Model compiled with torch.compile()")
    except Exception as e:
        print(f"torch.compile() failed: {e}, continuing without compilation")

# Optimizer
optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE)

# --- 4. Helper Function for Discounted Rewards and Advantage ---


def calculate_returns_and_advantages(rewards, values, gamma):
    """
    Calculates discounted returns (Gt) and the Advantage (At = Gt - V(s)).

    FIXED: Replaced tensor slicing (values[::-1]) with torch.flip to avoid
    'ValueError: step must be greater than zero' when slicing tensors.
    """
    returns = []
    R = 0

    # Reverse the lists/tensors for backward pass calculation
    reversed_rewards = rewards[::-1]
    # Use torch.flip for safe and explicit tensor reversal
    reversed_values = values.flip(dims=[0])

    # Loop backwards through rewards and values
    for r, v in zip(reversed_rewards, reversed_values):
        # r is a Python float/int, v is a PyTorch tensor element
        R = r + gamma * R
        returns.insert(0, R)

    returns = torch.tensor(returns, dtype=torch.float32).to(device)

    # Calculate Advantage: A(s,a) = G(s,a) - V(s)
    # The value network (Critic) gives V(s), which serves as the baseline.
    # Note: .detach() prevents gradients from flowing through the value prediction when calculating the Actor loss.
    advantages = returns - values.detach()

    # Normalize advantages for stability
    if len(advantages) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-9)
    else:
        advantages = torch.zeros_like(advantages)  # Handle single-step cases

    return returns, advantages


# --- 5. Training Loop ---

print("\nStarting Actor-Critic (A2C) simulation loop...")

# Open log file
log_file = open(LOG_FILE, "w")
log_file.write(
    "Episode,Steps,TotalReward,ActorLoss,CriticLoss,MeanValue,ValueStd,MeanReturn,CriticGrad\n"
)

for episode in range(NUM_EPISODES):
    # Data storage for the episode
    episode_rewards = []
    episode_log_probs = []
    episode_values = []
    episode_entropy = []

    # Reset environment
    observation, info = env.reset()
    terminated = False
    truncated = False

    steps = 0
    while not terminated and not truncated:
        image_tensor = torch.from_numpy(observation).permute(3, 0, 1, 2).float()
        input_tensor = image_tensor.to(device) / 255.0

        # Forward pass with mixed precision
        with torch.cuda.amp.autocast(enabled=USE_MIXED_PRECISION, dtype=dtype):
            action_logits, value = agent(input_tensor)

        # My initial intuition was to just argmax, but for RL we don't do that -
        # we sample the distribution. If we always pick the max (argmax), then
        # learning stalls.
        action_dist = Categorical(probs=F.softmax(action_logits.float(), dim=-1))
        action = action_dist.sample()

        # --- Store Data ---
        episode_log_probs.append(action_dist.log_prob(action))
        episode_values.append(value.squeeze(0))  # Store predicted value V(s)

        # Calculate and store entropy (for exploration bonus)
        episode_entropy.append(action_dist.entropy())

        # --- Step Environment ---
        observation, reward, terminated, truncated, info = env.step(action.item())

        # Clip reward for stability (prevents extreme values from destabilizing critic)
        clipped_reward = np.clip(reward, -REWARD_CLIP, REWARD_CLIP)
        episode_rewards.append(clipped_reward)

        steps += 1

        # Render the agent's grayscale view
        # We display the normalized 84x84 image, converting it back to 8-bit for cv2
        display_frame = observation[-1, :, :, :]
        display_frame = np.interp(
            display_frame, (display_frame.min(), display_frame.max()), (0, 255)
        ).astype(np.uint8)
        cv2.imshow("Agent View (84x84 Grayscale)", display_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            # Use sys.exit() to stop execution immediately
            sys.exit("Simulation stopped by user.")

    # --- Policy and Value Update (A2C Core) ---

    if not episode_rewards:
        print(f"Episode {episode + 1:4d} completed with 0 steps. Skipping update.")
        observation, info = env.reset()
        continue

    # 1. Prepare collected tensors
    log_probs = torch.cat(episode_log_probs)
    values = torch.cat(episode_values)
    entropy = torch.cat(episode_entropy).mean()

    # 2. Calculate discounted returns and advantages (using the Value Baseline)
    # The 'values' passed here are the Critic's predictions V(s)
    returns, advantages = calculate_returns_and_advantages(
        episode_rewards, values, GAMMA
    )

    # 3-5. Compute losses with mixed precision
    with torch.cuda.amp.autocast(enabled=USE_MIXED_PRECISION, dtype=dtype):
        # Actor loss: maximize (log_prob * Advantage)
        actor_loss = -(log_probs * advantages).mean()
        # Critic loss: Huber loss for robustness
        critic_loss = F.smooth_l1_loss(values, returns) * VALUE_COEFF
        # Total loss with entropy bonus
        total_loss = actor_loss + critic_loss - (entropy * ENTROPY_COEFF)

    optimizer.zero_grad()

    # Backward pass with gradient scaling (if using float16)
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()

    # Diagnostic: Check if gradients are flowing
    critic_grad_norm = (
        agent.critic.weight.grad.norm().item()
        if agent.critic.weight.grad is not None
        else 0.0
    )

    # Gradient clipping
    if scaler is not None:
        scaler.unscale_(optimizer)  # Unscale before clipping
    torch.nn.utils.clip_grad_norm_(agent.parameters(), 10.0)

    # Optimizer step with scaling
    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()

    # --- Logging ---
    total_reward = sum(episode_rewards)

    # Diagnostic metrics
    mean_value = values.mean().item()
    mean_return = returns.mean().item()
    value_std = values.std().item()

    # Print to console
    log_line = (
        f"Episode {episode + 1:4d} | Steps: {steps:3d} | Total Reward: {total_reward:6.2f} | "
        f"Actor Loss: {actor_loss.item():.4f} | Critic Loss: {critic_loss.item():.4f} | "
        f"Value: {mean_value:6.2f}±{value_std:.2f} | Return: {mean_return:6.2f} | "
        f"CriticGrad: {critic_grad_norm:.4f}"
    )
    print(log_line)

    # Write to log file (CSV format)
    log_file.write(
        f"{episode + 1},{steps},{total_reward:.2f},{actor_loss.item():.4f},"
        f"{critic_loss.item():.4f},{mean_value:.2f},{value_std:.2f},"
        f"{mean_return:.2f},{critic_grad_norm:.4f}\n"
    )
    log_file.flush()  # Ensure it writes immediately

    # Reset the environment for the next episode
    observation, info = env.reset()

# --- Cleanup ---
log_file.close()
env.close()
cv2.destroyAllWindows()
print(f"\nSimulation complete. Environment closed. Logs saved to {LOG_FILE}")
