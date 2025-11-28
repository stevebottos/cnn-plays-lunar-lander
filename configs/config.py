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
    GAMMA: float = 0.99
    GAE_LAMBDA: float = 0.95
    VALUE_COEFF: float = 0.5
    ENTROPY_COEFF: float = 0.01
    CLIP_EPSILON: float = 0.2
    TARGET_KL: None | float = None
    PPO_EPOCHS: int = 4
    BATCH_SIZE: int = 128
    NUM_ROUNDS: int = 1000
    GRU_LEARNING_RATE: None | float = None
    CHECKPOINT_PATH: None | str = None
    CHECKPOINT_FREQUENCY: int = 50
    MAX_EPISODE_STEPS: int = 250
    NUM_STEPS_PER_ROLLOUT: int = 1024
