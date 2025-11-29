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


class TinyCNN(nn.Module):
    def __init__(self, num_actions=4, *args, **kwargs):
        super(TinyCNN, self).__init__()

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

        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.temporal_pool1(x)

        identity = self.skip2(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = x + identity
        x = self.temporal_pool2(x)

        x = x.squeeze(2)

        identity = self.skip3(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = x + identity

        identity = self.skip4(x)
        x = self.relu(self.bn4(self.conv4(x)))
        x = x + identity

        identity = self.skip5(x)
        x = self.relu(self.bn5(self.conv5(x)))
        x = x + identity

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
            nn.BatchNorm3d(32),
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
            nn.BatchNorm3d(64),
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
            nn.BatchNorm3d(128),
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
            nn.BatchNorm3d(256),
        )

        self.skip1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=1, stride=(2, 2, 2), bias=False),
            nn.BatchNorm3d(32),
        )
        self.skip2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=1, stride=(2, 2, 2), bias=False),
            nn.BatchNorm3d(64),
        )
        self.skip3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=1, stride=(2, 2, 2), bias=False),
            nn.BatchNorm3d(128),
        )
        self.skip4 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=1, stride=(2, 2, 2), bias=False),
            nn.BatchNorm3d(256),
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
            elif isinstance(m, nn.BatchNorm3d):
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


class GatedMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_gate = nn.Linear(input_dim, hidden_dim)
        self.cell_gate = nn.Linear(input_dim, hidden_dim)
        self.output_gate = nn.Linear(input_dim, hidden_dim)

        for layer in [self.input_gate, self.cell_gate, self.output_gate]:
            nn.init.orthogonal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        i = torch.sigmoid(self.input_gate(x))
        g = torch.tanh(self.cell_gate(x))
        o = torch.sigmoid(self.output_gate(x))
        c = i * g
        h = o * torch.tanh(c)
        return h


class TinyCNNv2Gated(nn.Module):
    def __init__(self, num_actions=4, gated_hidden=768, *args, **kwargs):
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
            nn.BatchNorm3d(32),
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
            nn.BatchNorm3d(64),
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
            nn.BatchNorm3d(128),
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
            nn.BatchNorm3d(256),
        )

        self.skip1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=1, stride=(2, 2, 2), bias=False),
            nn.BatchNorm3d(32),
        )
        self.skip2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=1, stride=(2, 2, 2), bias=False),
            nn.BatchNorm3d(64),
        )
        self.skip3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=1, stride=(2, 2, 2), bias=False),
            nn.BatchNorm3d(128),
        )
        self.skip4 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=1, stride=(2, 2, 2), bias=False),
            nn.BatchNorm3d(256),
        )

        self.act = nn.SiLU()
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.gated_mlp = GatedMLP(256, gated_hidden)
        self.actor = nn.Linear(gated_hidden, num_actions)
        self.critic = nn.Linear(gated_hidden, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm3d):
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
        x = self.gated_mlp(x)

        action_logits = self.actor(x)
        value = self.critic(x)

        return action_logits, value


class TinyCNNv2LSTM(nn.Module):
    def __init__(self, num_actions=4, lstm_hidden=256, *args, **kwargs):
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
            nn.BatchNorm3d(32),
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
            nn.BatchNorm3d(64),
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
            nn.BatchNorm3d(128),
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
            nn.BatchNorm3d(256),
        )

        self.skip1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=1, stride=(2, 2, 2), bias=False),
            nn.BatchNorm3d(32),
        )
        self.skip2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=1, stride=(2, 2, 2), bias=False),
            nn.BatchNorm3d(64),
        )
        self.skip3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=1, stride=(2, 2, 2), bias=False),
            nn.BatchNorm3d(128),
        )
        self.skip4 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=1, stride=(2, 2, 2), bias=False),
            nn.BatchNorm3d(256),
        )

        self.act = nn.SiLU()
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True
        )

        self.actor = nn.Linear(lstm_hidden, num_actions)
        self.critic = nn.Linear(lstm_hidden, 1)
        self.hidden = None

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        for name, param in self.lstm.named_parameters():
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

    def forward(self, x, hidden=None):
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
        x = x.unsqueeze(1)

        lstm_out, new_hidden = self.lstm(x, hidden)
        lstm_out = lstm_out.squeeze(1)

        action_logits = self.actor(lstm_out)
        value = self.critic(lstm_out)

        return action_logits, value, new_hidden


class TinyCNNv3(nn.Module):
    def __init__(self, num_actions=4, *args, **kwargs):
        super(TinyCNNv3, self).__init__()

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

        self.flatten = nn.Flatten()
        self.feature_extractor_output_size = 256 * 4 * 4

        self.relu = nn.ReLU(inplace=True)

        self.actor = nn.Linear(self.feature_extractor_output_size, num_actions)
        self.critic = nn.Linear(self.feature_extractor_output_size, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.temporal_pool1(x)

        identity = self.skip2(x)
        x = self.bn2(self.conv2(x))
        x = self.relu(x + identity)
        x = self.temporal_pool2(x)

        x = x.squeeze(2)

        identity = self.skip3(x)
        x = self.bn3(self.conv3(x))
        x = self.relu(x + identity)

        identity = self.skip4(x)
        x = self.bn4(self.conv4(x))
        x = self.relu(x + identity)

        identity = self.skip5(x)
        x = self.bn5(self.conv5(x))
        x = self.relu(x + identity)

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
