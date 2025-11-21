from dataclasses import dataclass


@dataclass
class TrainingConfig:
    USE_MIXED_PRECISION: bool
    USE_TORCH_COMPILE: bool
    LEARNING_RATE: float
    GAMMA: float
    GAE_LAMBDA: float
    NUM_ROUNDS: int
    VALUE_COEFF: float
    ENTROPY_COEFF: float
    REWARD_CLIP: float
    CLIP_EPSILON: float
    BATCH_SIZE: int
    PPO_EPOCHS: int
    COLLECT_EPISODES: int
    IMAGE_SIZE: int
    NUM_FRAMES: int
    MAX_EPISODE_STEPS: int
    env_name: str
    model_name: str
