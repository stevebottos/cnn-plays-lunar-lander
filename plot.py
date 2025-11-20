import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the training log
df = pd.read_csv('training_log_ppo.txt')

# Create figure with subplots
fig, axes = plt.subplots(3, 2, figsize=(15, 12))
fig.suptitle('PPO Training Metrics', fontsize=16)

# 1. Total Reward over Episodes
ax = axes[0, 0]
ax.plot(df['Episode'], df['TotalReward'], alpha=0.3, label='Raw')
# Moving average
window = 20
if len(df) >= window:
    ma = df['TotalReward'].rolling(window=window).mean()
    ax.plot(df['Episode'], ma, linewidth=2, label=f'{window}-episode MA')
ax.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Success threshold')
ax.set_xlabel('Episode')
ax.set_ylabel('Total Reward')
ax.set_title('Reward Progress')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. Actor and Critic Loss
ax = axes[0, 1]
ax.plot(df['Episode'], df['ActorLoss'], label='Actor Loss', alpha=0.7)
ax.plot(df['Episode'], df['CriticLoss'], label='Critic Loss', alpha=0.7)
ax.set_xlabel('Episode')
ax.set_ylabel('Loss')
ax.set_title('Training Losses')
ax.legend()
ax.grid(True, alpha=0.3)

# 3. Value Function (Mean ± Std)
ax = axes[1, 0]
ax.plot(df['Episode'], df['MeanValue'], label='Mean Value', color='blue')
ax.fill_between(df['Episode'],
                 df['MeanValue'] - df['ValueStd'],
                 df['MeanValue'] + df['ValueStd'],
                 alpha=0.3, color='blue', label='±1 Std')
ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
ax.set_xlabel('Episode')
ax.set_ylabel('Value')
ax.set_title('Critic Value Predictions')
ax.legend()
ax.grid(True, alpha=0.3)

# 4. Value Std (Variance over time)
ax = axes[1, 1]
ax.plot(df['Episode'], df['ValueStd'], color='orange')
ax.set_xlabel('Episode')
ax.set_ylabel('Value Standard Deviation')
ax.set_title('Value Function Variance (Higher = More Discrimination)')
ax.grid(True, alpha=0.3)

# 5. Clip Fraction
ax = axes[2, 0]
ax.plot(df['Episode'], df['ClipFraction'], color='purple', alpha=0.7)
ax.axhline(y=0.1, color='g', linestyle='--', alpha=0.5, label='Ideal min (0.1)')
ax.axhline(y=0.3, color='g', linestyle='--', alpha=0.5, label='Ideal max (0.3)')
ax.set_xlabel('Episode')
ax.set_ylabel('Clip Fraction')
ax.set_title('PPO Clipping Rate')
ax.legend()
ax.grid(True, alpha=0.3)

# 6. Steps per Episode
ax = axes[2, 1]
ax.plot(df['Episode'], df['Steps'], color='brown', alpha=0.5)
if len(df) >= window:
    ma_steps = df['Steps'].rolling(window=window).mean()
    ax.plot(df['Episode'], ma_steps, linewidth=2, label=f'{window}-episode MA')
ax.set_xlabel('Episode')
ax.set_ylabel('Steps')
ax.set_title('Episode Length')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ppo_training_plots.png', dpi=150)
print("Plot saved as ppo_training_plots.png")

# Print summary statistics
print("\n=== Training Summary ===")
print(f"Total episodes: {len(df)}")
print(f"\nReward Statistics:")
print(f"  Best reward: {df['TotalReward'].max():.2f} (episode {df.loc[df['TotalReward'].idxmax(), 'Episode']:.0f})")
print(f"  Worst reward: {df['TotalReward'].min():.2f}")
print(f"  Mean reward: {df['TotalReward'].mean():.2f}")
print(f"  Latest 20 episodes mean: {df['TotalReward'].tail(20).mean():.2f}")

positive_episodes = df[df['TotalReward'] > 0]
print(f"\nPositive episodes: {len(positive_episodes)} ({100*len(positive_episodes)/len(df):.1f}%)")
if len(positive_episodes) > 0:
    print(f"  First positive at episode: {positive_episodes['Episode'].min():.0f}")

print(f"\nValue Function:")
print(f"  Mean value: {df['MeanValue'].iloc[-1]:.2f}")
print(f"  Value std: {df['ValueStd'].iloc[-1]:.2f}")
if df['ValueStd'].iloc[-1] < 0.1:
    print("  WARNING: Value variance collapsed! Critic not discriminating states.")

print(f"\nClip Fraction:")
print(f"  Latest: {df['ClipFraction'].iloc[-1]:.4f}")
print(f"  Mean (last 20): {df['ClipFraction'].tail(20).mean():.4f}")
if df['ClipFraction'].tail(20).mean() < 0.05:
    print("  WARNING: Very low clip fraction - policy barely updating!")

print(f"\nCritic Loss:")
print(f"  Latest: {df['CriticLoss'].iloc[-1]:.4f}")
print(f"  Mean (last 20): {df['CriticLoss'].tail(20).mean():.4f}")

plt.show()
