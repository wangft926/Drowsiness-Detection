import os
import sys
import time
import torch
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from torchvision.models import mobilenet_v3_small
from torch.nn.init import xavier_uniform_

# 自定义模块（请确保这些模块已实现）
from config_parser import parse_file


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
        # geo_input = geo_input.transpose(1, 2)  # (B, T, F)
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
output_dir = os.path.join('/kaggle/working', f'output_{timestamp}')
os.makedirs(output_dir, exist_ok=True)


# -----------------------------
# ✅ Step 2: 重定向 print 输出到日志文件
# -----------------------------
class Logger:
    def __init__(self, file):
        self.terminal = sys.stdout
        self.log = open(file, "w", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


sys.stdout = Logger(os.path.join(output_dir, "train.log"))

# -----------------------------
# ✅ Step 3: 加载配置 & 数据
# -----------------------------
config_file = parse_file('config.ini')
cnn_lstm_model = DualInputModel(config_file).to("cuda" if torch.cuda.is_available() else "cpu")
print(cnn_lstm_model)

# -----------------------------
# ✅ Step: 只取前 n 个 .npz 文件，再按比例划分
# -----------------------------
preprocessed_dir = 'dataset_npy'
max_files = config_file.get('max_files', 2)  # 你可以放在 config.ini 中设置 max_files=20

# 获取所有 .npz 文件，并排序以保证顺序一致（可选）
all_files = sorted([f for f in os.listdir(preprocessed_dir) if f.endswith('.npz')])

# 取前 n 个文件
selected_files = all_files[:max_files]
total_files = len(selected_files)

# 根据比例划分数据集
train_ratio = config_file['train_test_split']
train_size = int(total_files * train_ratio)
train_files = selected_files[:train_size]
val_files = selected_files[train_size:]

print(f"🔢 总共选取 {total_files} 个视频文件（前 {max_files} 个）")
print(f"✅ 训练集: {len(train_files)} 个文件")
print(f"🔍 验证集: {len(val_files)} 个文件")

# -----------------------------
# ✅ Step 4: 自定义 Lazy Dataset（支持逐个加载）
# -----------------------------
from torch.utils.data import Dataset
import numpy as np
import os
import torch


class CachedNPZDataset(Dataset):
    def __init__(self, file_list, root_dir, mean=None, std=None):
        self.root_dir = root_dir
        self.file_list = file_list
        self.samples = []

        # 缓存所有数据
        for file_idx, filename in enumerate(file_list):
            path = os.path.join(root_dir, filename)
            print(f"🧠 Loading {filename} into memory...")
            with np.load(path) as data:
                features = data['get_features'].astype(np.float32)
                images = data['face_images'].astype(np.float32)
                labels = data['labels']

                for i in range(len(labels)):
                    self.samples.append((
                        features[i],
                        images[i],
                        labels[i]
                    ))

        # 归一化参数（可选）
        if mean is not None and std is not None:
            self.mean = torch.tensor(mean, dtype=torch.float32)
            self.std = torch.tensor(std + 1e-8, dtype=torch.float32)
        else:
            self.mean = None
            self.std = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        geo, face, label = self.samples[idx]

        if self.mean is not None and self.std is not None:
            geo = (geo - self.mean) / self.std

        geo_tensor = torch.tensor(geo).float()
        face_tensor = torch.tensor(face).float().permute(0, 3, 1, 2)  # (T, C, H, W)
        label_tensor = torch.tensor(label).long()

        return geo_tensor, face_tensor, label_tensor


# 加载 mean & std（来自 dataset_wft.py 输出的 feature_stats.npz）
stats = np.load(os.path.join("dataset_npy_mean", "feature_stats.npz"))
global_mean = stats['mean']
global_std = stats['std']
# -----------------------------
# ✅ Step 5: 创建 Dataloader
# -----------------------------
batch_size = config_file['batch_size']
train_dataset = CachedNPZDataset(train_files, preprocessed_dir, mean=global_mean, std=global_std)
val_dataset = CachedNPZDataset(val_files, preprocessed_dir, mean=global_mean, std=global_std)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# 加载 class_idx
with open(config_file['class_idx_file'], 'rb') as obj:
    class_idx = pickle.load(obj)

# -----------------------------
# ✅ Step 6: 定义损失函数、优化器、学习率调度器
# -----------------------------
label_smoothing = config_file.get('label_smoothing', 0.1)
criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
optimizer = optim.AdamW(cnn_lstm_model.parameters(), lr=3e-4, weight_decay=1e-5)
num_epochs = config_file['num_epochs']
num_training_steps = len(train_loader) * num_epochs
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=3e-4, total_steps=num_training_steps,
    pct_start=0.3, anneal_strategy='cos'
)

alpha = 1
gamma = 2
training_history = []

# -----------------------------
# ✅ Step 7: 开始训练循环
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn_lstm_model.train()
y_true, y_pred = [], []
for epoch in range(num_epochs):
    total_loss = 0
    total_correct = 0

    for batch_geo, batch_face, batch_labels in train_loader:
        batch_geo = batch_geo.to(device)
        batch_face = batch_face.to(device)
        batch_labels = batch_labels.to(device)

        optimizer.zero_grad()
        outputs = cnn_lstm_model(batch_geo, batch_face)
        # loss = criterion(outputs, batch_labels)
        #
        # pt = torch.exp(-loss.detach())
        # focal_loss = alpha * (1 - pt) ** gamma * loss
        # focal_loss.backward()
        loss = criterion(outputs, batch_labels)
        loss.backward()
        # 可选：梯度裁剪（对 LSTM 很有用）
        # torch.nn.utils.clip_grad_norm_(cnn_lstm_model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        predicted = torch.argmax(outputs, dim=1)
        total_correct += (predicted == batch_labels).sum().item()
        total_loss += loss.item()

    current_lr = optimizer.param_groups[0]['lr']
    avg_loss = total_loss / len(train_loader)
    train_acc = total_correct / len(train_dataset) * 100

    # 验证阶段
    cnn_lstm_model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for inputs_geo, inputs_face, labels in val_loader:
            inputs_geo = inputs_geo.to(device)
            inputs_face = inputs_face.to(device)
            labels = labels.to(device)

            outputs = cnn_lstm_model(inputs_geo, inputs_face)
            # predicted = torch.argmax(outputs, dim=1)
            probs = torch.softmax(outputs, dim=1)
            predicted = torch.argmax(probs, dim=1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)
            y_true.extend(labels.tolist())
            y_pred.extend(predicted.tolist())

    val_acc = val_correct / val_total * 100 if val_total > 0 else 0

    training_history.append({
        'epoch': epoch + 1,
        'loss': avg_loss,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'learning_rate': current_lr
    })

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.8f}, '
          f'Train Acc: {train_acc:.4f}%, Val Acc: {val_acc:.4f}%')

# -----------------------------
# ✅ Step 8: 保存模型
# -----------------------------
model_save_path = os.path.join(output_dir, 'model-2G.pth')
torch.save(cnn_lstm_model.state_dict(), model_save_path)
print(f"Step 8: 保存模型 Model saved to {model_save_path}")

# -----------------------------
# ✅ Step 9: 在整个验证集上收集预测结果
# -----------------------------
# cnn_lstm_model.eval()
# y_true, y_pred = [], []
#
# with torch.no_grad():
#     for inputs_geo, inputs_face, labels in val_loader:
#         inputs_geo = inputs_geo.to(device)
#         inputs_face = inputs_face.to(device)
#         labels = labels.to(device)
#
#         outputs = cnn_lstm_model(inputs_geo, inputs_face)
#         probs = torch.softmax(outputs, dim=1)
#         predicted = torch.argmax(probs, dim=1)
#
#         y_true.extend(labels.tolist())
#         y_pred.extend(predicted.tolist())
pd.DataFrame(training_history).to_csv(os.path.join(output_dir, "training_epoch_metrics.csv"), index=False)

report_data = []
report = classification_report(y_true, y_pred, target_names=class_idx.keys(), output_dict=True)
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

cnf_mat = confusion_matrix(y_true, y_pred)
confusion_df = pd.DataFrame(cnf_mat,
                            index=[f"Actual_{cls}" for cls in class_idx.keys()],
                            columns=[f"Pred_{cls}" for cls in class_idx.keys()])
confusion_df.to_csv(os.path.join(output_dir, "confusion_matrix.csv"))

# -----------------------------
# ✅ Step 10: 绘图并保存
# -----------------------------
plt.figure(figsize=(10, 6))
plt.plot([x['loss'] for x in training_history], label='Train Loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'loss_graph.png'))

plt.figure(figsize=(10, 6))
plt.plot([x['learning_rate'] for x in training_history], label='Learning Rate')
plt.title('Learning Rate')
plt.xlabel('Epoch')
plt.ylabel('LR')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'lr_graph.png'))

plt.figure(figsize=(10, 6))
plt.plot([x['train_acc'] for x in training_history], label='Train Accuracy')
plt.plot([x['val_acc'] for x in training_history], label='Val Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'acc_graph.png'))

# Precision, Recall, F1 Score
intersection_matrix = np.vstack([
    precision_recall_fscore_support(y_true, y_pred)[i][:config_file['num_classes']]
    for i in range(3)
])
plt.figure(figsize=(12, 6))
cax1 = plt.matshow(intersection_matrix, cmap=plt.cm.Blues)
for i in range(3):
    for j in range(config_file['num_classes']):
        c = intersection_matrix[i, j]
        plt.text(j, i, str(round(c, 4)), va='center', ha='center')
plt.yticks(np.arange(3), ['Precision', 'Recall', 'F1-Score'])
plt.xticks(np.arange(config_file['num_classes']), list(class_idx.keys()))
plt.colorbar(cax1)
plt.title('Precision, Recall, and F1-Score per Class')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Precision_recall_f1score.png'))

# Confusion Matrix
plt.figure(figsize=(12, 10))
cax = plt.matshow(cnf_mat, cmap=plt.cm.Blues)
for i in range(config_file['num_classes']):
    for j in range(config_file['num_classes']):
        c = cnf_mat[j, i]
        plt.text(i, j, str(int(c)), va='center', ha='center')
plt.yticks(np.arange(config_file['num_classes']), list(class_idx.keys()))
plt.xticks(np.arange(config_file['num_classes']), list(class_idx.keys()))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.colorbar(cax)
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Confusion matrix.png'))

# -----------------------------
# ✅ Step 11: 打印最终分类报告
# -----------------------------
print("Step 11: 打印最终分类报告 Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_idx.keys(), digits=4))
