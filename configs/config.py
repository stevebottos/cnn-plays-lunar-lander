from dataclasses import dataclass


@dataclass
class TrainingConfig:
    # System Configuration
    USE_MIXED_PRECISION: bool
    USE_TORCH_COMPILE: bool

    # Model Configuration
    model_name: str

    # PPO Hyperparameters
    LEARNING_RATE: float
    GAMMA: float
    GAE_LAMBDA: float
    VALUE_COEFF: float
    ENTROPY_COEFF: float
    CLIP_EPSILON: float
    PPO_EPOCHS: int
    BATCH_SIZE: int
    NUM_ROUNDS: int
