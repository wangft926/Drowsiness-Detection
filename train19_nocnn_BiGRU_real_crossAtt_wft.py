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

# 自定义模块（请确保这些模块已实现）
from config_parser import parse_file
from customdata import CustomDataset
from dataset import preprocess_data

import torch.nn.functional as F
import cv2
import torch
from torch.utils.data import Dataset
import random
from torch.optim.lr_scheduler import CosineAnnealingLR

from torchvision.models import mobilenet_v3_small

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small
from torch.nn.init import xavier_uniform_


# 相比train18，使用 OneCycleLR（更推荐）
class CustomDataset_new(Dataset):
    def __init__(self, geo_data, face_data, labels, augment=True):
        self.geo_data = geo_data  # (N, num_features, window_size)
        self.face_data = face_data  # (N, window_size, H, W, C)
        self.labels = labels  # (N,)
        self.augment = augment  # 是否启用增强

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        geo = self.geo_data[idx]  # (num_features, window_size)
        face = self.face_data[idx]  # (window_size, H, W, C)
        label = self.labels[idx]

        # ✅ 数据增强部分（每次读取样本时动态处理）
        if self.augment:
            face = self.random_augment(face)

        # 转为 tensor
        face = torch.from_numpy(face).float().permute(0, 3, 1, 2)  # (T, C, H, W)
        geo = torch.from_numpy(geo).float()
        label = torch.tensor(label).long()

        return geo, face, label

    def random_augment(self, frames):
        """
        对一串视频帧做简单增强（亮度扰动 + 噪声 + 随机裁剪）
        :param frames: (T, H, W, C) 的 numpy 数组
        :return: 增强后的 frames
        """
        augmented_frames = []
        for img in frames:
            # 1️⃣ 随机亮度调整
            if random.random() < 0.5:
                brightness_factor = random.uniform(0.8, 1.2)
                img = cv2.convertScaleAbs(img, alpha=brightness_factor, beta=0)

            # 2️⃣ 添加高斯噪声
            if random.random() < 0.5:
                noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
                img = cv2.add(img, noise)

            # 3️⃣ 随机裁剪并恢复尺寸
            if random.random() < 0.5:
                h, w = img.shape[:2]
                scale = random.uniform(0.8, 1.0)
                new_h, new_w = int(h * scale), int(w * scale)
                top = random.randint(0, h - new_h)
                left = random.randint(0, w - new_w)
                img = img[top:top + new_h, left:left + new_w]
                img = cv2.resize(img, (w, h))  # 恢复原尺寸

            augmented_frames.append(img)

        return np.array(augmented_frames)


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


# -----------------------------
# ✅ Step 1: 创建带时间戳的输出目录
# -----------------------------
timestamp = time.strftime("%Y%m%d_%H%M%S")
# output_dir = f"/output_{timestamp}"
output_dir = os.path.join('/kaggle/working', f'output_{timestamp}')
os.makedirs(output_dir, exist_ok=True)
# -----------------------------
# ✅ Step 2: 重定向 print 输出到日志文件
# -----------------------------
import sys


class Logger:
    def __init__(self, file):
        self.terminal = sys.stdout
        self.log = open(file, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


sys.stdout = Logger(os.path.join(output_dir, "train.log"))

# -----------------------------
# ✅ Step 3: 加载配置和数据
# -----------------------------
config_file = parse_file('config.ini')
cnn_lstm_model = DualInputModel(config_file).to("cuda" if torch.cuda.is_available() else "cpu")
print(cnn_lstm_model)

# if config_file['use_existing_preprocessesd_data']:
windows_arr = np.load(config_file['preprocessed_windows_data'])
face_windows_arr = np.load(config_file['preprocessed_face_windows_data'])
labels_array = np.load(config_file['preprocessed_labels_data'])
with open(config_file['class_idx_file'], 'rb') as obj:
    class_idx = pickle.load(obj)
# else:
#     class_idx, windows_arr, face_windows_arr, labels_array = preprocess_data(config_file)
#     with open(config_file['class_idx_file'], 'wb') as obj:
#         pickle.dump(class_idx, obj)

# 打乱顺序
indices = np.random.permutation(len(windows_arr)).astype(np.int64)
windows_shuffled = windows_arr[indices]
face_shuffled = face_windows_arr[indices]
labels_shuffled = labels_array[indices]

print('Classes present - ', Counter(labels_shuffled))

# -----------------------------
# ✅ Step 4: 数据划分（先在 CPU 上进行）
# -----------------------------
X_train_geo_np, X_test_geo_np, X_train_face_np, X_test_face_np, y_train_np, y_test_np = train_test_split(
    windows_shuffled, face_shuffled, labels_shuffled,
    test_size=config_file['train_test_split'],
    random_state=42
)

# 删除原始大数组以节省内存
del windows_arr, face_windows_arr, labels_array, windows_shuffled, face_shuffled, labels_shuffled
torch.cuda.empty_cache()

# 转换为 Tensor（仅在需要时移动到 GPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# X_train_geo = torch.tensor(X_train_geo_np, dtype=torch.float32)
# X_train_face = torch.tensor(X_train_face_np, dtype=torch.float32)
# y_train = torch.tensor(y_train_np, dtype=torch.float32)
# X_test_geo = torch.tensor(X_test_geo_np, dtype=torch.float32)
# X_test_face = torch.tensor(X_test_face_np, dtype=torch.float32)
# y_test = torch.tensor(y_test_np, dtype=torch.float32)

train_dataset = CustomDataset(X_train_geo_np, X_train_face_np, y_train_np)
test_dataset = CustomDataset(X_test_geo_np, X_test_face_np, y_test_np)

# train_dataset = CustomDataset(X_train_geo_np, X_train_face_np, y_train_np)
# test_dataset = CustomDataset(X_test_geo_np, X_test_face_np, y_test_np)

batch_size = config_file['batch_size']
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# -----------------------------
# ✅ Step 5: 定义损失函数、优化器、学习率调度器
# -----------------------------
# criterion = nn.CrossEntropyLoss()
label_smoothing = config_file.get('label_smoothing', 0.1)
criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
# optimizer = optim.Adam(cnn_lstm_model.parameters(), lr=config_file['lr'])
# optimizer = optim.Adam(cnn_lstm_model.parameters(), lr=config_file['lr'], weight_decay=1e-4)  # 添加 L2 正则化项
optimizer = optim.AdamW(cnn_lstm_model.parameters(), lr=3e-4, weight_decay=1e-5)#weight_decay=1e-5 就是L2
# num_epochs = config_file['num_epochs']
num_epochs = 100
num_training_steps = len(train_loader) * num_epochs
#scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=3e-4, total_steps=num_training_steps,
    pct_start=0.3, anneal_strategy='cos'    # 使用 cosine 衰减
)

alpha = 1
gamma = 2
training_history = []

# -----------------------------
# ✅ Step 6: 开始训练循环
# -----------------------------
for epoch in range(num_epochs):
    cnn_lstm_model.train()
    total_loss = 0
    total_correct = 0

    for batch_inputs_geo, batch_inputs_face, batch_labels in train_loader:
        batch_inputs_geo = batch_inputs_geo.to(device)
        batch_inputs_face = batch_inputs_face.to(device)
        batch_labels = batch_labels.to(device)

        optimizer.zero_grad()
        outputs = cnn_lstm_model(batch_inputs_geo, batch_inputs_face)
        loss = criterion(outputs, batch_labels)

        pt = torch.exp(-loss.detach())
        F_loss = alpha * (1 - pt) ** gamma * loss
        F_loss.backward()
        optimizer.step()
        scheduler.step()

        predicted = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
        total_correct += (predicted == batch_labels).sum().item()
        total_loss += F_loss.item()

    current_lr = optimizer.param_groups[0]['lr']
    avg_loss = total_loss / len(train_loader)
    accuracy = total_correct / len(train_dataset) * 100

    training_history.append({
        'epoch': epoch + 1,
        'loss': avg_loss,
        'accuracy': accuracy,
        'learning_rate': current_lr
    })

    print(f'Epoch [{epoch + 1}/{num_epochs}], Learning Rate: {current_lr:.6f}')
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.8f}, Accuracy: {accuracy:.4f}%')

# -----------------------------
# ✅ Step 7: 保存模型
# -----------------------------
model_save_path = os.path.join(output_dir, 'model-2G.pth')
torch.save(cnn_lstm_model.state_dict(), model_save_path)
print(f"Step 7: 保存模型 Model saved to {model_save_path}")

# -----------------------------
# ✅ Step 8: 推理与评估
# -----------------------------
cnn_lstm_model.eval()
y_true = []
y_pred = []

with torch.no_grad():
    for inputs_geo, inputs_face, labels in DataLoader(test_dataset, batch_size=batch_size):
        inputs_geo = inputs_geo.to(device)
        inputs_face = inputs_face.to(device)
        outputs = cnn_lstm_model(inputs_geo, inputs_face)
        predicted = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
        y_true.extend(labels.numpy())
        y_pred.extend(predicted.cpu().numpy())

report = classification_report(y_true, y_pred, target_names=class_idx.keys(), output_dict=True)
cnf_mat = confusion_matrix(y_true, y_pred)
print(f"Step 8: 推理与评估")
# -----------------------------
# ✅ Step 9: 保存指标到 CSV
# -----------------------------
import pandas as pd

# 1. 训练历史
pd.DataFrame(training_history).to_csv(os.path.join(output_dir, "training_epoch_metrics.csv"), index=False)

# 2. 分类报告
report_data = []
for cls_name, values in report.items():
    if cls_name in class_idx or cls_name in ['macro avg', 'weighted avg']:
        report_data.append({
            'class': cls_name,
            'precision': values.get('precision', None),
            'recall': values.get('recall', None),
            'f1_score': values.get('f1-score', None),
            'support': values.get('support', None)
        })
pd.DataFrame(report_data).to_csv(os.path.join(output_dir, "classification_report.csv"), index=False)

# 3. 混淆矩阵
confusion_df = pd.DataFrame(cnf_mat,
                            index=[f"Actual_{cls}" for cls in class_idx.keys()],
                            columns=[f"Pred_{cls}" for cls in class_idx.keys()])
confusion_df.to_csv(os.path.join(output_dir, "confusion_matrix.csv"))
print(f"Step 9: 保存指标到 CSV")
# -----------------------------
# ✅ Step 10: 绘图并保存
# -----------------------------
plt.figure(0)
plt.plot([x['loss'] for x in training_history])
plt.title('Average loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.tight_layout()  # 自动调整子图参数，防止重叠
plt.savefig(os.path.join(output_dir, 'loss_graph.png'))

plt.figure(1)
plt.plot([x['learning_rate'] for x in training_history])
plt.title('Learning rate')
plt.xlabel('Epoch')
plt.ylabel('Lr')
plt.tight_layout()  # 自动调整子图参数，防止重叠
plt.savefig(os.path.join(output_dir, 'lr_graph.png'))

plt.figure(2)
plt.plot([x['accuracy'] for x in training_history])
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.tight_layout()  # 自动调整子图参数，防止重叠
plt.savefig(os.path.join(output_dir, 'acc_graph.png'))

# 精确率、召回率、F1-score 图
intersection_matrix = np.vstack([
    precision_recall_fscore_support(y_true, y_pred)[i] for i in range(3)
])

plt.figure(figsize=(10, 10))
cax1 = plt.matshow(intersection_matrix[:config_file['num_classes'], :], cmap=plt.cm.Blues)
for i in range(config_file['num_classes']):
    for j in range(config_file['num_classes']):
        c = intersection_matrix[j, i]
        plt.text(i, j, str(round(c, 4)), va='center', ha='center')
plt.yticks(np.arange(config_file['num_classes']), ['Precision', 'Recall', 'F1score'])
plt.xticks(np.arange(config_file['num_classes']),
           [f"{cls}\n{round(cnt, 4)}" for cls, cnt in zip(list(class_idx.keys()), intersection_matrix[-1, :])])
plt.tight_layout()  # 自动调整布局
# 添加颜色刻度条（关键修改）
plt.colorbar(cax1)
plt.savefig(os.path.join(output_dir, 'Precision_recall_f1score.png'))

# 混淆矩阵图
plt.figure(figsize=(14, 10))
# plt.matshow(cnf_mat, cmap=plt.cm.Blues)
cax = plt.matshow(cnf_mat, cmap=plt.cm.Blues)  # 使用 cax 接收返回值用于 colorbar
for i in range(config_file['num_classes']):
    for j in range(config_file['num_classes']):
        c = cnf_mat[j, i]
        plt.text(i, j, str(round(c, 4)), va='center', ha='center')
plt.yticks(np.arange(config_file['num_classes']), list(class_idx.keys()))
plt.xticks(np.arange(config_file['num_classes']),
           [f"{cls}\n{cnt}" for cls, cnt in zip(list(class_idx.keys()), cnf_mat.sum(axis=1))])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()  # 自动调整布局
# 添加颜色刻度条（关键修改）
plt.colorbar(cax)
plt.savefig(os.path.join(output_dir, 'Confusion matrix.png'))
print(f"Step 10: 绘图并保存")
# -----------------------------
# ✅ Step 11: 打印最终分类报告
# -----------------------------
print("Step 11: 打印最终分类报告 Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_idx.keys(), digits=4))
