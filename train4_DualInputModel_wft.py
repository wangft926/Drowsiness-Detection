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

# 自定义模块（请确保这些模块已实现）
from model_dual import DualInputModel
from config_parser import parse_file
from dataset import preprocess_data
from customdata import CustomDataset

# 几何特征+图片特征，王凤桐
# -----------------------------
# ✅ Step 1: 创建带时间戳的输出目录
# -----------------------------
timestamp = time.strftime("%Y%m%d_%H%M%S")
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
cnn_lstm_model = DualInputModel(config_file)
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

indices = np.random.permutation(len(windows_arr)).astype(np.int64)
print("indices shape:", indices.shape)
print("indices dtype:", indices.dtype)
print("windows_arr shape:", windows_arr.shape)
print("face_windows_arr shape:", face_windows_arr.shape)
print("labels_array shape:", labels_array.shape)
print("indices shape:", indices.shape)
cnn_inp_shuffled = windows_arr[indices]
face_inp_shuffled = face_windows_arr[indices]
cnn_labels_shuffled = labels_array[indices]

print('Classes present - ', Counter(labels_array))

# -----------------------------
# ✅ Step 4: 数据转换 & 划分训练测试集
# -----------------------------
# 转换为 Tensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
geo_tensor = torch.tensor(cnn_inp_shuffled, dtype=torch.float32, device=device)
face_tensor = torch.tensor(face_inp_shuffled, dtype=torch.float32, device=device)
label_tensor = torch.tensor(cnn_labels_shuffled, dtype=torch.long, device=device)

# 数据划分
X_train_geo, X_test_geo, X_train_face, X_test_face, y_train, y_test = train_test_split(
    geo_tensor, face_tensor, label_tensor,
    test_size=config_file['train_test_split'],
    random_state=42
)

# 使用双输入 Dataset
dataset = CustomDataset(X_train_geo, X_train_face, y_train)
batch_size = config_file['batch_size']
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
# -----------------------------
# ✅ Step 5: 定义损失函数、优化器、学习率调度器
# -----------------------------
# criterion = nn.CrossEntropyLoss()
label_smoothing = config_file.get('label_smoothing', 0.1)
criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
optimizer = optim.Adam(cnn_lstm_model.parameters(), lr=config_file['lr'])

# num_epochs = config_file['num_epochs']
num_epochs = 80
num_training_steps = len(data_loader) * num_epochs
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=0.01, total_steps=num_training_steps,
    pct_start=0.3, anneal_strategy='linear'
)

alpha = 1
gamma = 2
lr_li = []
accuracy_li = []
avg_loss_li = []

# 用于保存每轮训练指标
training_history = []

# -----------------------------
# ✅ Step 6: 开始训练循环
# -----------------------------
for epoch in range(num_epochs):
    cnn_lstm_model.train()
    total_loss = 0
    total_correct = 0

    for batch_inputs_geo, batch_inputs_face, batch_labels in data_loader:
        optimizer.zero_grad()
        print(f"face_input shape: {batch_inputs_face.shape}")
        outputs = cnn_lstm_model(batch_inputs_geo, batch_inputs_face)
        loss = criterion(outputs, batch_labels)

        pt = torch.exp(-loss)
        F_loss = alpha * (1 - pt) ** gamma * loss
        total_loss += F_loss.item()
        F_loss.backward()
        optimizer.step()
        scheduler.step()

        predicted = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
        total_correct += (predicted == batch_labels).sum().item()

    current_lr = optimizer.param_groups[0]['lr']
    avg_loss = total_loss / len(data_loader)
    accuracy = total_correct / len(dataset) * 100

    # 记录本轮指标
    training_history.append({
        'epoch': epoch + 1,
        'loss': avg_loss,
        'accuracy': accuracy,
        'learning_rate': current_lr
    })

    print(f'Epoch [{epoch + 1}/{num_epochs}], Learning Rate: {current_lr:.6f}')
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

# -----------------------------
# ✅ Step 7: 保存模型到 output_dir
# -----------------------------
model_save_path = os.path.join(output_dir, 'model-2G.pth')
torch.save(cnn_lstm_model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# -----------------------------
# ✅ Step 8: 推理与评估
# -----------------------------
model = DualInputModel(config_file)
model.load_state_dict(torch.load(model_save_path, weights_only=True))
model.eval()
with torch.no_grad():
    pred = model(X_test_geo, X_test_face)  # 👈 双输入
    predicted = torch.argmax(torch.softmax(pred, dim=1), dim=1)

# 分类报告
report = classification_report(y_test, predicted, target_names=class_idx.keys(), output_dict=True)

# 混淆矩阵
cnf_mat = confusion_matrix(y_test, predicted)

# -----------------------------
# ✅ Step 9: 保存所有指标到 CSV
# -----------------------------

import pandas as pd

# 1️⃣ 保存每轮训练指标
train_df = pd.DataFrame(training_history)
train_csv_path = os.path.join(output_dir, "training_epoch_metrics.csv")
train_df.to_csv(train_csv_path, index=False)
print(f"Epoch metrics saved to {train_csv_path}")

# 2️⃣ 保存分类报告（precision, recall, f1-score, support）
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
report_df = pd.DataFrame(report_data)
cls_csv_path = os.path.join(output_dir, "classification_report.csv")
report_df.to_csv(cls_csv_path, index=False)
print(f"Classification report saved to {cls_csv_path}")

# 3️⃣ 保存混淆矩阵
confusion_df = pd.DataFrame(cnf_mat,
                            index=[f"Actual_{cls}" for cls in class_idx.keys()],
                            columns=[f"Pred_{cls}" for cls in class_idx.keys()])
confusion_csv_path = os.path.join(output_dir, "confusion_matrix.csv")
confusion_df.to_csv(confusion_csv_path)
print(f"Confusion matrix saved to {confusion_csv_path}")

# -----------------------------
# ✅ Step 10: 绘图并保存到 output_dir
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
    precision_recall_fscore_support(y_test, predicted)[i] for i in range(3)
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

# -----------------------------
# ✅ Step 11: 打印最终分类报告
# -----------------------------
print("Classification Report:")
print(classification_report(y_test, predicted, target_names=class_idx.keys(), digits=4))
