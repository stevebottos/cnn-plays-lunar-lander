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

        self.embed_dim = 256
        self.seq_len = 256

        self.conv3d_stem = nn.Sequential(
            nn.Conv3d(1, self.embed_dim, kernel_size=(4, 16, 16), stride=(4, 16, 16)),
        )
        self.pos_embed = torch.nn.Parameter(
            torch.randn(1, self.seq_len, self.embed_dim) * 0.02
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=8,
            dim_feedforward=256,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=8)

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
    def __init__(self, num_actions=4, *args, **kwargs):
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

        self.actor = nn.Linear(256, num_actions)
        self.critic = nn.Linear(256, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Orthogonal initialization for actor and critic layers
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def forward(self, x):
        # First 3D convolutions to process temporal + spatial info
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.temporal_pool1(x)

        identity = self.skip2(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = x + identity  # Skip connection
        x = self.temporal_pool2(x)

        # Remove temporal dimension
        x = x.squeeze(2)

        # 2D convolutions with skip connections
        identity = self.skip3(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = x + identity

        identity = self.skip4(x)
        x = self.relu(self.bn4(self.conv4(x)))
        x = x + identity

        identity = self.skip5(x)
        x = self.relu(self.bn5(self.conv5(x)))
        x = x + identity

        # Global pooling and final projection
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        action_logits = self.actor(x)
        value = self.critic(x)
        return action_logits, value


class TinyCNNv2(nn.Module):
    def __init__(self, num_actions=4, *args, **kwargs):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv3d(
                1,
                32,
                kernel_size=(3, 3, 3),
                stride=(2, 2, 2),
                padding=(1, 1, 1),
                bias=False,
            ),
            nn.GroupNorm(8, 32),  # Only change: BatchNorm3d → GroupNorm
        )
        self.block2 = nn.Sequential(
            nn.Conv3d(
                32,
                64,
                kernel_size=(3, 3, 3),
                stride=(2, 2, 2),
                padding=(1, 1, 1),
                bias=False,
            ),
            nn.GroupNorm(8, 64),
        )
        self.block3 = nn.Sequential(
            nn.Conv3d(
                64,
                128,
                kernel_size=(3, 3, 3),
                stride=(2, 2, 2),
                padding=(1, 1, 1),
                bias=False,
            ),
            nn.GroupNorm(16, 128),
        )
        self.block4 = nn.Sequential(
            nn.Conv3d(
                128,
                256,
                kernel_size=(3, 3, 3),
                stride=(2, 2, 2),
                padding=(1, 1, 1),
                bias=False,
            ),
            nn.GroupNorm(32, 256),
        )

        self.skip1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=1, stride=(2, 2, 2), bias=False),
            nn.GroupNorm(8, 32),
        )
        self.skip2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=1, stride=(2, 2, 2), bias=False),
            nn.GroupNorm(8, 64),
        )
        self.skip3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=1, stride=(2, 2, 2), bias=False),
            nn.GroupNorm(16, 128),
        )
        self.skip4 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=1, stride=(2, 2, 2), bias=False),
            nn.GroupNorm(32, 256),
        )

        self.act = nn.SiLU()
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.actor = nn.Linear(256, num_actions)
        self.critic = nn.Linear(256, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.GroupNorm, nn.BatchNorm3d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def forward(self, x):
        identity = self.skip1(x)
        x = self.block1(x)
        x = self.act(x + identity)

        identity = self.skip2(x)
        x = self.block2(x)
        x = self.act(x + identity)

        identity = self.skip3(x)
        x = self.block3(x)
        x = self.act(x + identity)

        identity = self.skip4(x)
        x = self.block4(x)
        x = self.act(x + identity)

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        action_logits = self.actor(x)
        value = self.critic(x)

        return action_logits, value


class TinyCNNv3(nn.Module):
    def __init__(self, num_actions=4, *args, **kwargs):
        super(TinyCNNv3, self).__init__()

        # Initial conv to reduce temporal dimension
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

        # Changed: Replaced AdaptiveAvgPool2d with Flatten
        self.flatten = nn.Flatten()
        self.feature_extractor_output_size = (
            256 * 4 * 4
        )  # 256 channels * 4x4 spatial dimensions

        self.relu = nn.ReLU(inplace=True)

        # Updated linear layers to match the new flattened output size
        self.actor = nn.Linear(self.feature_extractor_output_size, num_actions)
        self.critic = nn.Linear(self.feature_extractor_output_size, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Orthogonal initialization for actor and critic layers
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def forward(self, x):
        # First 3D convolutions to process temporal + spatial info
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.temporal_pool1(x)

        identity = self.skip2(x)
        x = self.bn2(self.conv2(x))
        x = self.relu(x + identity)  # Skip connection
        x = self.temporal_pool2(x)

        # Remove temporal dimension
        x = x.squeeze(2)

        # 2D convolutions with skip connections
        identity = self.skip3(x)
        x = self.bn3(self.conv3(x))
        x = self.relu(x + identity)

        identity = self.skip4(x)
        x = self.bn4(self.conv4(x))
        x = self.relu(x + identity)

        identity = self.skip5(x)
        x = self.bn5(self.conv5(x))
        x = self.relu(x + identity)

        # Flatten instead of Global pooling
        x = self.flatten(x)

        action_logits = self.actor(x)
        value = self.critic(x)
        return action_logits, value


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on: {device}\n")

    num_actions = 4
    num_frames = 16
    img_size = 128
    dummy_input = torch.randn(1, 1, 16, img_size, img_size).to(device)
    model = Conv3DTransformerNet(num_actions=num_actions).to(device)
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
