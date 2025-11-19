# type: ignore
import torch
import torch.nn as nn
import numpy as np
import torchvision


class Squeeze(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.squeeze(2)


class Conv3dResNet(nn.Module):
    def __init__(self, num_actions=4, *args, **kwargs) -> None:
        super().__init__()

        self.backbone = torchvision.models.resnet18(weights=None)
        self.backbone.fc = nn.Identity()

        self.backbone.conv1 = nn.Sequential(
            nn.Conv3d(
                1,
                64,
                kernel_size=(16, 7, 7),
                stride=(16, 2, 2),
                padding=(0, 3, 3),
                bias=False,
            ),
            Squeeze(),
        )

        self.actor = nn.Linear(512, num_actions)
        self.critic = nn.Linear(512, 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def forward(self, x):
        x = self.backbone(x)
        action_logits = self.actor(x)
        value = self.critic(x)
        return action_logits, value


class Conv3DTransformerNet(nn.Module):
    def __init__(self, num_actions, num_frames=16):
        super().__init__()
        self.num_actions = num_actions
        self.num_frames = num_frames

        self.conv3d_stem = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(8, 32, 32), stride=(4, 32, 32)),
        )

        self.seq_len = 147
        self.embed_dim = 64

        self.pos_embed = nn.Parameter(
            torch.randn(1, self.seq_len, self.embed_dim) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=4,
            dim_feedforward=256,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.norm = nn.LayerNorm(self.embed_dim)
        self.actor = nn.Linear(self.embed_dim, num_actions)
        self.critic = nn.Linear(self.embed_dim, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def forward(self, x):
        x = self.conv3d_stem(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.norm(x)
        action_logits = self.actor(x)
        value = self.critic(x)
        return action_logits, value


class TinyCNN(nn.Module):
    """
    Lightweight CNN feature encoder for RL with skip connections.
    Input: (batch, 1, 16, 224, 224) - grayscale, 16 frames, 224x224
    Output: (batch, feature_dim) - flattened feature vector
    """

    def __init__(self, *args, **kwargs):
        super(TinyCNN, self).__init__()

        feature_dim = 512
        # Initial conv to reduce temporal dimension
        # (1, 16, 224, 224) -> (32, 16, 112, 112)
        self.conv1 = nn.Conv3d(
            1, 32, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=(1, 1, 1)
        )
        self.bn1 = nn.BatchNorm3d(32)
        self.temporal_pool1 = nn.MaxPool3d(kernel_size=(4, 1, 1), stride=(4, 1, 1))

        self.conv2 = nn.Conv3d(
            32, 64, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)
        )
        self.bn2 = nn.BatchNorm3d(64)
        self.skip2 = nn.Conv3d(32, 64, kernel_size=1, stride=(1, 2, 2))
        self.temporal_pool2 = nn.MaxPool3d(kernel_size=(4, 1, 1), stride=(4, 1, 1))

        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.skip3 = nn.Conv2d(64, 128, kernel_size=1, stride=2)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.skip4 = nn.Conv2d(128, 256, kernel_size=1, stride=2)

        self.conv5 = nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.skip5 = nn.Conv2d(256, 256, kernel_size=1, stride=2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.relu = nn.ReLU(inplace=True)

        self.actor = nn.Linear(256, 4)
        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        # First 3D convolutions to process temporal + spatial info
        x = self.relu(self.bn1(self.conv1(x)))  # (batch, 32, 16, 112, 112)
        x = self.temporal_pool1(x)  # (batch, 32, 4, 112, 112)

        identity = self.skip2(x)
        x = self.relu(self.bn2(self.conv2(x)))  # (batch, 64, 4, 56, 56)
        x = x + identity  # Skip connection
        x = self.temporal_pool2(x)  # (batch, 64, 1, 56, 56)

        # Remove temporal dimension
        x = x.squeeze(2)  # (batch, 64, 56, 56)

        # 2D convolutions with skip connections
        identity = self.skip3(x)
        x = self.relu(self.bn3(self.conv3(x)))  # (batch, 128, 28, 28)
        x = x + identity

        identity = self.skip4(x)
        x = self.relu(self.bn4(self.conv4(x)))  # (batch, 256, 14, 14)
        x = x + identity

        identity = self.skip5(x)
        x = self.relu(self.bn5(self.conv5(x)))  # (batch, 256, 7, 7)
        x = x + identity

        # Global pooling and final projection
        x = self.global_pool(x)  # (batch, 256, 1, 1)
        x = x.view(x.size(0), -1)  # (batch, 256)
        action_logits = self.actor(x)
        value = self.critic(x)
        return action_logits, value


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on: {device}\n")

    num_actions = 4
    num_frames = 16
    img_size = 224

    print("=" * 60)
    print("Testing Conv3dResNet")
    print("=" * 60)

    model = TinyCNN(num_actions=num_actions).to(device)

    dummy_input = torch.randn(1, 1, 16, img_size, img_size).to(device)
    print(f"Input shape: {dummy_input.shape}")

    action_logits, value = model(dummy_input)
    print(f"Action logits shape: {action_logits.shape}")
    print(f"Value shape: {value.shape}")
    print(f"Action logits sample: {action_logits[0]}")
    print(f"Value sample: {value[0]}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)
