# dual_model_a.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class DualInputModel(nn.Module):
    def __init__(self, config):
        super(DualInputModel, self).__init__()

        # 几何特征分支（Dilated CNN）
        self.cnn_geo = nn.Sequential(
            nn.Conv1d(config['num_features'], 32, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=2, dilation=2),  # dilation=2
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # 图像帧分支（3D CNN）
        self.cnn_img = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2))
        )
        #Version2-最开始的特征降维-img_bottleneck，效果很好
        self.img_bottleneck = nn.Sequential(
            nn.Linear(32 * 32 * 32, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )

        # LSTM 分支
        self.lstm_geo = nn.LSTM(64, 64, batch_first=True)
        self.lstm_img = nn.LSTM(128, 64, batch_first=True)

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, config['num_classes'])
        )

    def forward(self, geo_input, face_input):
        B, T = face_input.shape[0], face_input.shape[1]

        # 几何特征分支
        geo_out = self.cnn_geo(geo_input)  # (B, C, T)
        geo_out = geo_out.transpose(1, 2)  # (B, T, C)
        geo_out, _ = self.lstm_geo(geo_out)  # (B, T, 64)

        # 图像分支
        face_input = face_input.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
        face_out = self.cnn_img(face_input)  # (B, C', T, H', W')

        # ✅ 关键修复点：使用 reshape 替代 view
        face_out = face_out.reshape(B, T, -1)  # (B, T, 32768)

        # 确保张量连续性再调用 view
        face_out = self.img_bottleneck(face_out.contiguous().view(B * T, -1)).view(B, T, -1)  # (B, T, 128)
        face_out, _ = self.lstm_img(face_out)  # (B, T, 64)

        # 合并最后一层
        combined = torch.cat([geo_out[:, -1, :], face_out[:, -1, :]], dim=1)  # (B, 128)
        out = self.classifier(combined)
        return out
