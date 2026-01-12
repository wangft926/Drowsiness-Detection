import torch
import os
import pickle
import time
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from torch.nn.init import xavier_uniform_
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v3_small

# 自定义模块（请确保这些模块已实现）
from config_parser import parse_file
from customdata import CustomDataset


# wft实验全集
# 相比train18，使用 OneCycleLR（更推荐）

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=2):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_k = nn.LayerNorm(embed_dim)
        self.norm_v = nn.LayerNorm(embed_dim)
        self.proj_out = nn.Linear(embed_dim, embed_dim)
        # ✅ 添加 dropout 层（记得在这加！）
        self.dropout = nn.Dropout(0.1)  # 可以根据需要调整 dropout 比例

    def forward(self, query, key, value):
        """
        Args:
            query: (B, Q, D)
            key:   (B, T, D)
            value: (B, T, D)
        Returns:
            fused: (B, D)
        """
        q = self.norm_q(query)
        k = self.norm_k(key)
        v = self.norm_v(value)

        # # ✅ 正确调用方式：位置参数传入 q, k, v
        attn_output, _ = self.attn(q, k, v)

        # 全局平均池化得到融合向量
        fused = torch.mean(attn_output, dim=1)
        return self.proj_out(fused)
        # attn_output, _ = self.attn(q, k, v)
        # attn_output = self.dropout(attn_output)
        #
        # # ✅ 残差连接：attn_output + query
        # residual = attn_output + query
        #
        # # 全局平均池化
        # fused = torch.mean(residual, dim=1)
        # return self.proj_out(fused)


class FrameFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = mobilenet_v3_small(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])  # 去掉最后的分类层
        self.proj = nn.Linear(576, 128)  # MobileNetV3-Small 输出是 576

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)  # 合并 batch 和 time
        features = self.feature_extractor(x)  # (B*T, 576, 1, 1)
        features = features.view(B * T, -1)  # (B*T, 576)
        features = self.proj(features)  # (B*T, 128)
        features = features.view(B, T, -1)  # (B, T, 128)
        return features


class GatedFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        gate_input = torch.cat([x1, x2], dim=-1)  # (B, 2D)
        gate = self.gate(gate_input)  # (B, D)
        return gate * x1 + (1 - gate) * x2  # (B, D)


class DualInputModel(nn.Module):
    def __init__(self, config):
        super(DualInputModel, self).__init__()

        # ----------------------------
        # 几何特征分支：BiGRU
        # ----------------------------
        self.gru_geo = nn.GRU(
            input_size=config['num_features'],
            hidden_size=128,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        self.norm_geo = nn.LayerNorm(256)
        self.proj_geo = nn.Linear(256, 128)  # BiGRU 输出 256 → 128

        # ----------------------------
        # 图像特征分支：MobileNetV3 提取帧级特征
        # ----------------------------
        self.cnn_img = FrameFeatureExtractor()
        self.gru_img = nn.GRU(
            input_size=128,
            hidden_size=128,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        self.norm_img = nn.LayerNorm(256)  # GRU 输出是 (B, T, 256)

        # Linear 投影
        self.proj_img = nn.Linear(256, 128)
        # ----------------------------
        # 双向 Cross Attention 模块
        # ----------------------------
        # 使用 learnable queries
        self.learnable_queries = nn.Parameter(torch.randn(1, 1, 128))  # (1, 1, 128)

        self.cross_attn_geo_to_img = CrossAttention(embed_dim=128)
        self.cross_attn_img_to_geo = CrossAttention(embed_dim=128)

        # Gated Fusion
        self.gated_fusion = GatedFusion(dim=128)

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.5),
            nn.Linear(64, config['num_classes'])
        )

        # 参数初始化
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, geo_input, face_input):
        B, T = face_input.shape[0], face_input.shape[1]

        # ----------------------------
        # 几何特征处理：BiGRU
        # ----------------------------
        geo_input = geo_input.transpose(1, 2)  # (B, T, F)
        gru_geo_out, _ = self.gru_geo(geo_input)  # (B, T, 256)
        gru_geo_out = self.norm_geo(gru_geo_out)  # ✅ LayerNorm on 256-dim
        geo_lstm = self.proj_geo(gru_geo_out)  # (B, T, 128)

        # ----------------------------
        # 图像特征处理：MobileNet 提取每帧特征
        # ----------------------------
        face_out = self.cnn_img(face_input)  # (B, T, 128)

        # 加入 BiGRU 捕捉帧间动态变化
        gru_img_out, _ = self.gru_img(face_out)  # (B, T, 256)
        gru_img_out = self.norm_img(gru_img_out)  # ✅ LayerNorm on 256-dim
        face_out = self.proj_img(gru_img_out)  # (B, T, 128)
        # ----------------------------
        # 双向 Cross Attention
        # ----------------------------
        queries = self.learnable_queries.expand(B, -1, -1)  # (B, 1, 128)

        # 几何 → 图像 注意力
        combined_geo = self.cross_attn_geo_to_img(
            query=queries,  # (B, 1, 128)
            key=face_out,  # (B, T, 128)
            value=face_out  # (B, T, 128)
        )

        # 图像 → 几何 注意力
        combined_img = self.cross_attn_img_to_geo(
            query=queries,  # (B, 1, 128)
            key=geo_lstm,  # (B, T, 128)
            value=geo_lstm  # (B, T, 128)
        )

        # ----------------------------
        # 融合两个方向的注意力输出
        # ----------------------------
        final = self.gated_fusion(combined_geo, combined_img)  # (B, 128)

        # ----------------------------
        # 分类输出
        # ----------------------------
        out = self.classifier(final)  # (B, num_classes)

        return out


from torchviz import make_dot

# 初始化模型
config = {'num_features': 5, 'num_classes': 3}  # 根据实际情况设置
model = DualInputModel(config)

x = torch.randn(1, 10)
y = model(x)
dot = make_dot(y, params=dict(list(model.named_parameters())))
dot.render("model", format="png")
