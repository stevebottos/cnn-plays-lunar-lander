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


class TemporalResNet(nn.Module):
    def __init__(self, num_actions=4, num_frames=16, embed_dim=512, num_heads=8, num_layers=4, *args, **kwargs):
        super().__init__()
        self.num_frames = num_frames
        self.embed_dim = embed_dim

        self.backbone = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Identity()

        pretrained_conv1_weight = self.backbone.conv1.weight.data
        grayscale_weight = pretrained_conv1_weight.mean(dim=1, keepdim=True)

        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone.conv1.weight.data = grayscale_weight

        if embed_dim != 512:
            self.feature_proj = nn.Linear(512, embed_dim)
        else:
            self.feature_proj = nn.Identity()

        self.pos_embed = nn.Parameter(torch.randn(1, num_frames, embed_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(embed_dim)
        self.actor = nn.Linear(embed_dim, num_actions)
        self.critic = nn.Linear(embed_dim, 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def forward(self, x):
        batch_size = x.size(0)

        x = x.squeeze(1)
        x = x.reshape(batch_size * self.num_frames, 1, x.size(2), x.size(3))

        features = self.backbone(x)
        features = features.reshape(batch_size, self.num_frames, -1)
        features = self.feature_proj(features)
        features = features + self.pos_embed
        features = self.transformer(features)
        features = features.mean(dim=1)
        features = self.norm(features)

        action_logits = self.actor(features)
        value = self.critic(features)

        return action_logits, value


class TemporalResNetGRU(nn.Module):
    def __init__(self, num_actions=4, num_frames=16, hidden_size=512, num_layers=2, *args, **kwargs):
        super().__init__()
        self.num_frames = num_frames
        self.hidden_size = hidden_size

        self.backbone = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Identity()

        pretrained_conv1_weight = self.backbone.conv1.weight.data
        grayscale_weight = pretrained_conv1_weight.mean(dim=1, keepdim=True)

        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone.conv1.weight.data = grayscale_weight

        self.gru = nn.GRU(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0
        )

        self.norm = nn.LayerNorm(hidden_size)
        self.actor = nn.Linear(hidden_size, num_actions)
        self.critic = nn.Linear(hidden_size, 1)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                nn.init.orthogonal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def forward(self, x):
        batch_size = x.size(0)

        x = x.squeeze(1)
        x = x.reshape(batch_size * self.num_frames, 1, x.size(2), x.size(3))

        features = self.backbone(x)
        features = features.reshape(batch_size, self.num_frames, -1)

        output, _ = self.gru(features)
        features = output[:, -1, :]

        features = self.norm(features)

        action_logits = self.actor(features)
        value = self.critic(features)

        return action_logits, value


class TemporalMobileNetGRU(nn.Module):
    def __init__(self, num_actions=4, num_frames=16, hidden_size=512, num_layers=2, *args, **kwargs):
        super().__init__()
        self.num_frames = num_frames
        self.hidden_size = hidden_size

        self.backbone = torchvision.models.mobilenet_v3_large(
            weights=torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
        )
        self.backbone.classifier = nn.Identity()

        pretrained_conv1_weight = self.backbone.features[0][0].weight.data
        grayscale_weight = pretrained_conv1_weight.mean(dim=1, keepdim=True)

        self.backbone.features[0][0] = nn.Conv2d(
            1, 16, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.backbone.features[0][0].weight.data = grayscale_weight

        self.gru = nn.GRU(
            input_size=960,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0
        )

        self.norm = nn.LayerNorm(hidden_size)
        self.actor = nn.Linear(hidden_size, num_actions)
        self.critic = nn.Linear(hidden_size, 1)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                nn.init.orthogonal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def forward(self, x):
        batch_size = x.size(0)

        x = x.squeeze(1)
        x = x.reshape(batch_size * self.num_frames, 1, x.size(2), x.size(3))

        features = self.backbone(x)
        features = features.reshape(batch_size, self.num_frames, -1)

        output, _ = self.gru(features)
        features = output[:, -1, :]

        features = self.norm(features)

        action_logits = self.actor(features)
        value = self.critic(features)

        return action_logits, value


class Conv3DTransformerNet(nn.Module):
    def __init__(self, num_actions, num_frames=16):
        super().__init__()
        self.num_actions = num_actions
        self.num_frames = num_frames

        self.embed_dim = 128
        self.seq_len = 256

        self.conv3d_stem = nn.Sequential(
            nn.Conv3d(1, self.embed_dim, kernel_size=(4, 16, 16), stride=(4, 16, 16)),
        )
        self.pos_embed = torch.nn.Parameter(
            torch.randn(1, self.seq_len, self.embed_dim) * 0.02
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=4,
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
