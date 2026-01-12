# dual_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class DualInputModel(nn.Module):
    def __init__(self, config):
        super(DualInputModel, self).__init__()

        # 几何特征分支
        #去掉池化层，仅用卷积控制感受野
        # self.cnn_geo = nn.Sequential(
        #     nn.Conv1d(config['num_features'], 32, kernel_size=3, padding=1),  # padding 保持 T
        #     nn.ReLU(),
        #     #nn.MaxPool1d(2),
        #     nn.Conv1d(32, 64, kernel_size=3, padding=1),  # padding 保持 T
        #     nn.ReLU(),
        #     #nn.MaxPool1d(2)
        # )
        # #方法二：Stride=1 的池化或卷积 + 降维
        # self.cnn_geo = nn.Sequential(
        #     nn.Conv1d(config['num_features'], 32, kernel_size=3, padding=1, stride=1),
        #     nn.BatchNorm1d(32),
        #     nn.ReLU(),
        #     nn.AvgPool1d(kernel_size=2, stride=1, padding=1),
        #     nn.Conv1d(32, 64, kernel_size=3, padding=1, stride=1),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(),
        #     nn.Dropout(0.3)
        # )

        #使用空洞卷积（Dilated Convolution）
        #时间维度不变，感受野更大（可以看到更多历史帧），非常适合时序建模任务
        # self.cnn_geo = nn.Sequential(
        #     nn.Conv1d(config['num_features'], 32, kernel_size=3, padding=1, dilation=1),
        #     nn.ReLU(),
        #     nn.Conv1d(32, 64, kernel_size=3, padding=2, dilation=2),  # dilation=2 扩展感受野
        #     nn.ReLU()
        # )
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
            nn.Conv3d(3, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1)),  # padding 保持 T 不变
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1)),  # 同样 padding
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2))
        )
        self.img_bottleneck = nn.Sequential(
            nn.Linear(32768, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )
        # LSTM 融合层
        cnn_geo_out_dim = 64
        cnn_img_out_dim = 32 * 32 * 32
        self.lstm = nn.LSTM(cnn_geo_out_dim + cnn_img_out_dim, 128, batch_first=True)

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, config['num_classes'])
        )

    def forward(self, geo_input, face_input):
        B, T = face_input.shape[0], face_input.shape[1]
        # 应为 (B, T, C, H, W)
        # 图像分支
        face_input = face_input.permute(0, 2, 1, 3, 4)

        face_out = self.cnn_img(face_input)
        face_out = face_out.permute(0, 2, 1, 3, 4)
        face_out = (face_out.contiguous().view(B, T, -1))
        # 几何特征分支
        geo_out = self.cnn_geo(geo_input)

        # # 如果时间维度被压缩，就插值
        # if geo_out.shape[-1] != T:
        #     geo_out = F.interpolate(geo_out, size=T, mode='linear', align_corners=True)
        #
        # # 🔍 可视化代码放在这里 ↓↓↓
        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(10, 3))
        # plt.plot(geo_input[0, 0].cpu().numpy(), label="Original Geo Feature")
        # plt.plot(geo_out[0, :, 0].cpu().detach().numpy(), '--', label="Upsampled Geo Feature")
        # plt.legend()
        # plt.title("Geo Feature Before and After Interpolation")
        # plt.show()

        geo_out = geo_out.transpose(1, 2)  # → (B, T, C)
        # 合并
        combined = torch.cat([geo_out, face_out], dim=2)

        # LSTM + 分类
        lstm_out, _ = self.lstm(combined)
        out = self.classifier(lstm_out[:, -1, :])
        return out
