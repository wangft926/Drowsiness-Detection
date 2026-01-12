import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Linear(dim * 2, dim)

    def forward(self, x1, x2):
        # x1: (B, D), x2: (B, D)
        gate_input = torch.cat([x1, x2], dim=-1)  # (B, 2D)
        gate = torch.sigmoid(self.gate(gate_input))  # (B, D)
        fused = gate * x1 + (1 - gate) * x2  # 加权融合
        return fused


class DualInputModel_Best(nn.Module):
    def __init__(self, config):
        super(DualInputModel_Best, self).__init__()

        # 几何特征分支
        self.cnn_geo = nn.Sequential(
            nn.Conv1d(config['num_features'], 32, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.lstm = nn.LSTM(64, 128, batch_first=True)

        # 图像帧分支
        self.cnn_img = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2))
        )
        self.proj = nn.Linear(32 * 32 * 32, 64)

        # Cross Attention
        self.cross_attn = nn.MultiheadAttention(embed_dim=64, num_heads=1)

        # Attention 到 LSTM 维度映射
        self.fusion_proj = nn.Linear(64, 128)

        # Gated Fusion
        self.gated_fusion = GatedFusion(dim=128)

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, config['num_classes'])
        )

    def forward(self, geo_input, face_input):
        B, T = face_input.shape[0], face_input.shape[1]

        # 几何特征处理
        geo_out = self.cnn_geo(geo_input).transpose(1, 2)  # (B, T, 64)
        lstm_out, _ = self.lstm(geo_out)  # (B, T, 128)
        geo_final = lstm_out[:, -1, :]  # (B, 128)

        # 图像特征处理
        face_input = face_input.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
        face_out = self.cnn_img(face_input)  # (B, C', T, H', W')
        face_out = face_out.view(B, T, -1)  # (B, T, 32768)
        face_out = self.proj(face_out)  # (B, T, 64)

        # Cross Attention: Geo → Image
        attn_output, _ = self.cross_attn(
            query=geo_out.transpose(0, 1),  # (T, B, 64)
            key=face_out.transpose(0, 1),  # (T, B, 64)
            value=face_out.transpose(0, 1)  # (T, B, 64)
        )
        combined = attn_output.mean(dim=0)  # (B, 64)

        # 投影到 LSTM 输出维度
        combined = self.fusion_proj(combined)  # (B, 128)

        # 使用门控机制融合
        final = self.gated_fusion(combined, geo_final)  # (B, 128)

        # 分类输出
        out = self.classifier(final)  # (B, num_classes)

        return out
