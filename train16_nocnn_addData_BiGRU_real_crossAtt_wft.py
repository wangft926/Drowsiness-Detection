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

import torch.nn.functional as F
import cv2
import torch
from torch.utils.data import Dataset
import random


# 有数据增强版本

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


def depthwise_separable_conv1d(in_channels, out_channels, kernel_size=3, padding=1):
    return nn.Sequential(
        nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels),
        nn.Conv1d(in_channels, out_channels, kernel_size=1)
    )


def depthwise_separable_conv3d(in_channels, out_channels, kernel_size=3, padding=1):
    return nn.Sequential(
        nn.Conv3d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels),
        nn.Conv3d(in_channels, out_channels, kernel_size=1)
    )


class DualInputModel(nn.Module):
    def __init__(self, config):
        super(DualInputModel, self).__init__()

        # ----------------------------
        # 几何特征分支：BiGRU
        # ----------------------------
        self.gru_geo = nn.GRU(
            input_size=config['num_features'],  # 如 EAR_L, EAR_R, MAR 等
            hidden_size=128,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        self.proj_geo = nn.Linear(256, 128)  # BiGRU 输出 256 → 128

        # ----------------------------
        # 图像特征分支：Conv3D + BiGRU
        # ----------------------------
        self.cnn_img = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2))
        )
        self.proj_img = nn.Linear(32 * 32 * 32, 64)
        self.gru_img = nn.GRU(64, 128, batch_first=True, bidirectional=True)
        self.proj_img_gru = nn.Linear(256, 128)

        # ----------------------------
        # 双向 Cross Attention 模块
        # ----------------------------
        self.attn_norm = nn.LayerNorm(64)

        # 投影到注意力空间
        self.proj_for_attn_geo = nn.Linear(128, 64)  # BiGRU 输出 128 → 注意力用 64
        self.proj_for_attn_img = nn.Linear(128, 64)

        # 几何 → 图像 注意力
        self.cross_attn_geo_to_img = nn.MultiheadAttention(embed_dim=64, num_heads=2, batch_first=False)

        # 图像 → 几何 注意力
        self.cross_attn_img_to_geo = nn.MultiheadAttention(embed_dim=64, num_heads=2, batch_first=False)

        # 可学习 Query 向量
        self.learnable_query = nn.Parameter(torch.randn(1, 1, 64))  # (1, B, 64)

        # 投影回融合空间
        self.fusion_proj_geo = nn.Linear(64, 128)
        self.fusion_proj_img = nn.Linear(64, 128)

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

        # 初始化 attention 权重
        for p in self.cross_attn_geo_to_img.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
        for p in self.cross_attn_img_to_geo.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, geo_input, face_input):
        B, T = face_input.shape[0], face_input.shape[1]
        # ----------------------------
        # 几何特征处理：BiGRU
        # ----------------------------
        # geo_input shape: (B, F, T) → 我们要转成 (B, T, F)
        geo_input = geo_input.transpose(1, 2)  # (B, 5, 16) → (B, 16, 5)
        gru_geo_out, _ = self.gru_geo(geo_input)  # 输入 shape: (B, T, F) → (B, T, 256)
        geo_lstm = self.proj_geo(gru_geo_out)  # (B, T, 128)

        # ----------------------------
        # 图像特征处理：Conv3D + BiGRU
        # ----------------------------
        face_input = face_input.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
        face_out = self.cnn_img(face_input)  # (B, 32, T, 32, 32)
        face_out = face_out.reshape(B, T, -1)  # (B, T, 32768)
        face_out = self.proj_img(face_out)  # (B, T, 64)
        face_out = F.normalize(face_out, dim=-1)

        gru_img_out, _ = self.gru_img(face_out)  # (B, T, 256)
        img_lstm = self.proj_img_gru(gru_img_out)  # (B, T, 128)

        # ----------------------------
        # 双向 Cross Attention
        # ----------------------------

        # 1. 几何 → 图像 注意力
        queries_geo = self.learnable_query.expand(-1, B, -1)  # (1, B, 64)

        geo_attn_input = self.proj_for_attn_geo(geo_lstm)  # (B, T, 64)
        keys_img = self.attn_norm(geo_attn_input.transpose(0, 1))  # (T, B, 64)
        values_img = keys_img

        attn_geo_to_img, _ = self.cross_attn_geo_to_img(
            query=queries_geo,
            key=keys_img,
            value=values_img
        )
        combined_geo = self.fusion_proj_geo(attn_geo_to_img.squeeze(0))  # (B, 128)

        # 2. 图像 → 几何 注意力
        queries_img = self.learnable_query.expand(-1, B, -1)  # (1, B, 64)

        img_attn_input = self.proj_for_attn_img(img_lstm)  # (B, T, 64)
        keys_geo = self.attn_norm(img_attn_input.transpose(0, 1))  # (T, B, 64)
        values_geo = keys_geo

        attn_img_to_geo, _ = self.cross_attn_img_to_geo(
            query=queries_img,
            key=keys_geo,
            value=values_geo
        )
        combined_img = self.fusion_proj_img(attn_img_to_geo.squeeze(0))  # (B, 128)

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

# train_dataset = CustomDataset(X_train_geo_np, X_train_face_np, y_train_np)
# test_dataset = CustomDataset(X_test_geo_np, X_test_face_np, y_test_np)

# train_dataset = CustomDataset(X_train_geo_np, X_train_face_np, y_train_np)
# test_dataset = CustomDataset(X_test_geo_np, X_test_face_np, y_test_np)
train_dataset = CustomDataset_new(X_train_geo_np, X_train_face_np, y_train_np, augment=True)
test_dataset = CustomDataset_new(X_test_geo_np, X_test_face_np, y_test_np)

batch_size = config_file['batch_size']
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# -----------------------------
# ✅ Step 5: 定义损失函数、优化器、学习率调度器
# -----------------------------
# criterion = nn.CrossEntropyLoss()
label_smoothing = config_file.get('label_smoothing', 0.1)
criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
# optimizer = optim.Adam(cnn_lstm_model.parameters(), lr=config_file['lr'])
optimizer = optim.Adam(cnn_lstm_model.parameters(), lr=config_file['lr'], weight_decay=1e-4)  # 添加 L2 正则化项

# num_epochs = config_file['num_epochs']
num_epochs = 80
num_training_steps = len(train_loader) * num_epochs
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=0.01, total_steps=num_training_steps,
    pct_start=0.3, anneal_strategy='linear'
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
plt.savefig(os.path.join(output_dir, 'loss_graph.png'))

plt.figure(1)
plt.plot([x['learning_rate'] for x in training_history])
plt.title('Learning rate')
plt.xlabel('Epoch')
plt.ylabel('Lr')
plt.savefig(os.path.join(output_dir, 'lr_graph.png'))

plt.figure(2)
plt.plot([x['accuracy'] for x in training_history])
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.savefig(os.path.join(output_dir, 'acc_graph.png'))

# 精确率、召回率、F1-score 图
intersection_matrix = np.vstack([
    precision_recall_fscore_support(y_true, y_pred)[i] for i in range(3)
])

plt.figure(figsize=(10, 10))
plt.matshow(intersection_matrix[:config_file['num_classes'], :], cmap=plt.cm.Blues)
for i in range(config_file['num_classes']):
    for j in range(config_file['num_classes']):
        c = intersection_matrix[j, i]
        plt.text(i, j, str(round(c, 4)), va='center', ha='center')
plt.yticks(np.arange(config_file['num_classes']), ['Precision', 'Recall', 'F1score'])
plt.xticks(np.arange(config_file['num_classes']),
           [f"{cls}\n{round(cnt, 4)}" for cls, cnt in zip(list(class_idx.keys()), intersection_matrix[-1, :])])
plt.savefig(os.path.join(output_dir, 'Precision_recall_f1score.png'))

# 混淆矩阵图
plt.figure(figsize=(12, 10))
plt.matshow(cnf_mat, cmap=plt.cm.Blues)
for i in range(config_file['num_classes']):
    for j in range(config_file['num_classes']):
        c = cnf_mat[j, i]
        plt.text(i, j, str(round(c, 4)), va='center', ha='center')
plt.yticks(np.arange(config_file['num_classes']), list(class_idx.keys()))
plt.xticks(np.arange(config_file['num_classes']),
           [f"{cls}\n{cnt}" for cls, cnt in zip(list(class_idx.keys()), cnf_mat.sum(axis=1))])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig(os.path.join(output_dir, 'Confusion matrix.png'))
print(f"Step 10: 绘图并保存")
# -----------------------------
# ✅ Step 11: 打印最终分类报告
# -----------------------------
print("Step 11: 打印最终分类报告 Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_idx.keys(), digits=4))
