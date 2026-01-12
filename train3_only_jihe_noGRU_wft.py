import os
import time
from collections import Counter
import pickle

import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report

from customdata import CustomDataset
# 自定义模块（请确保这些模块已实现）
from model import CNN_LSTM
from config_parser import parse_file
from dataset import preprocess_data

import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_


import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_

class DualInputModel(nn.Module):
    def __init__(self, config):
        super(DualInputModel, self).__init__()

        # ----------------------------
        # 几何特征分支：无时序建模
        # ----------------------------
        # 直接使用原始特征，不做时序建模
        self.norm_geo = nn.LayerNorm(config['num_features'])  # 对原始特征做归一化
        self.proj_geo = nn.Linear(config['num_features'], 128)  # (F → 128)

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
        #print("geo_input shape:", geo_input.shape)  # 确保输出是 (64, 16, 5)
        geo_input = geo_input.transpose(1, 2)  # 变成 (64, 5, 16) → (64, 16, 5)
        B, T, F = geo_input.shape  # 假设 geo_input 的形状是 (B, T, F)

        # ----------------------------
        # 几何特征处理：去掉 GRU
        # ----------------------------
        geo = self.norm_geo(geo_input)  # LayerNorm on features
        geo = self.proj_geo(geo)        # (B, T, 128)

        # 全局平均池化
        final = torch.mean(geo, dim=1)  # (B, 128)

        # ----------------------------
        # 分类输出
        # ----------------------------
        out = self.classifier(final)    # (B, num_classes)

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

train_dataset = CustomDataset(X_train_geo_np, X_train_face_np, y_train_np)
test_dataset = CustomDataset(X_test_geo_np, X_test_face_np, y_test_np)

batch_size = config_file['batch_size']
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# -----------------------------
# ✅ Step 5: 定义损失函数、优化器、学习率调度器
# -----------------------------
# criterion = nn.CrossEntropyLoss()
label_smoothing = config_file.get('label_smoothing', 0.1)
criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
optimizer = optim.Adam(cnn_lstm_model.parameters(), lr=config_file['lr'])

# num_epochs = config_file['num_epochs']
num_epochs = 100
num_training_steps = len(train_loader) * num_epochs
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=0.01, total_steps=num_training_steps,
    pct_start=0.3, anneal_strategy='cos'
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
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.6f}, Accuracy: {accuracy:.4f}%')

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
