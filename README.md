# Vision-Based PPO for Lunar Lander

Vision-only reinforcement learning agent for LunarLander-v3 using Proximal Policy Optimization.

## Architecture

**Model**: TemporalResNetGRU
- ResNet18 backbone (ImageNet pretrained) for per-frame feature extraction
- 2-layer GRU for temporal aggregation across 16-frame sequences
- Separate actor-critic heads

**Observations**: 16-frame grayscale stack (128×128 pixels)

**Environment**: 8 parallel AsyncVectorEnv instances

## Key Hyperparameters

```yaml
LEARNING_RATE: 0.00003
GAE_LAMBDA: 0.90
ENTROPY_COEFF: 0.008
CLIP_EPSILON: 0.15
TARGET_KL: 0.06
PPO_EPOCHS: 1
BATCH_SIZE: 128
NUM_STEPS_PER_ROLLOUT: 2048
MAX_EPISODE_STEPS: 1000
```

**Critical finding**: `GAE_LAMBDA=0.90` (down from default 0.95) was essential for training stability. Lower lambda reduces bias from distant rewards and prevents catastrophic forgetting in vision-based settings.

## Training

```bash
python train_v3.py --config configs/temporal_resnet_gru_11.yaml
```

Checkpoints saved to `checkpoints/` every N rounds (configurable via `CHECKPOINT_FREQUENCY`).

Training logs written to `run.jsonl` and tracked via MLflow.

## Evaluation

```bash
python gather_samples.py --checkpoint checkpoints/0000100.pt --num-episodes 50
```

Generates GIFs of episodes and computes average reward.

**Note**: Uses stochastic action sampling (not argmax) to match training evaluation. High policy entropy (~0.5) means greedy selection performs significantly worse.

## Performance

- **Solved threshold**: 200+ average reward
- **Achieved**: Consistent 180-220 range after ~750 rounds
- **Stability**: No boom-bust cycles with proper GAE_LAMBDA tuning

## Project Structure

```
train_v3.py              # Main PPO training loop
models.py                # Neural network architectures
gather_samples.py        # Checkpoint evaluation and GIF generation
configs/
  config.py              # TrainingConfig dataclass
  *.yaml                 # Experiment configurations
```

## Requirements

See `pyproject.toml` for dependencies. Primary requirements:
- PyTorch 2.0+
- Gymnasium
- MLflow
