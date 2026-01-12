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
from model import CNN_LSTM
from config_parser import parse_file


class CustomDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]


# -----------------------------
# 🟨 Step A: 定义 EarlyStopping 类
# -----------------------------
class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decreases."""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


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

# 实例化模型并打印结构
cnn_lstm_model = CNN_LSTM(config_file)
print(cnn_lstm_model)

# 加载预处理后的数据
windows_arr = np.load(config_file['preprocessed_windows_data'])
labels_array = np.load(config_file['preprocessed_labels_data'])

with open(config_file['class_idx_file'], 'rb') as obj:
    class_idx = pickle.load(obj)

# 打乱数据
indices = np.random.permutation(len(windows_arr))
cnn_inp_shuffled = windows_arr[indices]
cnn_labels_shuffled = labels_array[indices]

print('Classes present - ', Counter(labels_array))

# -----------------------------
# ✅ Step 4: 转换为 Tensor 并划分数据集
# -----------------------------
cnn_inp_shuffled = torch.tensor(cnn_inp_shuffled, dtype=torch.float32)
cnn_labels_shuffled = torch.tensor(cnn_labels_shuffled, dtype=torch.long)

# 划分测试集
X_train_full, X_test, y_train_full, y_test = train_test_split(
    cnn_inp_shuffled, cnn_labels_shuffled,
    test_size=config_file['train_test_split'],
    random_state=42
)

# 再从训练集中划分出验证集
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size=config_file.get('val_split', 0.1),
    random_state=42
)

# 构建 DataLoader
train_dataset = CustomDataset(X_train, y_train)
val_dataset = CustomDataset(X_val, y_val)
test_dataset = CustomDataset(X_test, y_test)

batch_size = config_file['batch_size']
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# -----------------------------
# ✅ Step 5: 定义损失函数、优化器、学习率调度器
# -----------------------------
# criterion = nn.CrossEntropyLoss()
label_smoothing = config_file.get('label_smoothing', 0.1)
criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
optimizer = optim.Adam(cnn_lstm_model.parameters(), lr=config_file['lr'])

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
# 🟨 Step B: 初始化早停机制
# -----------------------------
early_stopping = EarlyStopping(
    patience=config_file.get('patience', 5),
    verbose=True,
    path=os.path.join(output_dir, 'best_model.pth')
)

# -----------------------------
# ✅ Step 6: 开始训练循环（含早停）
# -----------------------------
for epoch in range(num_epochs):
    cnn_lstm_model.train()
    total_loss = 0
    total_correct = 0

    for batch_inputs, batch_labels in train_loader:
        optimizer.zero_grad()
        outputs = cnn_lstm_model(batch_inputs)
        loss = criterion(outputs, batch_labels)

        pt = torch.exp(-loss)
        focal_loss = alpha * (1 - pt) ** gamma * loss
        total_loss += focal_loss.item()
        focal_loss.backward()
        optimizer.step()
        scheduler.step()

        predicted = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
        total_correct += (predicted == batch_labels).sum().item()

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

    # 🟨 验证阶段
    cnn_lstm_model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_inputs, batch_labels in val_loader:
            outputs = cnn_lstm_model(batch_inputs)
            loss = criterion(outputs, batch_labels)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f'Validation Loss: {avg_val_loss:.4f}')

    # 🟨 调用 EarlyStopping
    early_stopping(avg_val_loss, cnn_lstm_model)
    if early_stopping.early_stop:
        print("Early stopping triggered.")
        break

# -----------------------------
# ✅ Step 7: 保存最终模型
# -----------------------------
model_save_path = os.path.join(output_dir, 'model-2G.pth')
torch.save(cnn_lstm_model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# -----------------------------
# ✅ Step 8: 推理与评估（使用最佳模型）
# -----------------------------
model = CNN_LSTM(config_file)
model.load_state_dict(torch.load(os.path.join(output_dir, 'best_model.pth')))
model.eval()

with torch.no_grad():
    pred = model(X_test)  # 使用测试集做最终评估
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
precision, recall, f1, _ = precision_recall_fscore_support(y_test, predicted, average=None)
metrics_matrix = np.vstack((precision, recall, f1))

plt.figure(figsize=(10, 5))
plt.imshow(metrics_matrix, cmap=plt.cm.Blues)
plt.colorbar()
plt.xticks(range(len(class_idx)), class_idx.keys())
plt.yticks(range(3), ['Precision', 'Recall', 'F1-Score'])
plt.title('Class-wise Metrics')
for i in range(metrics_matrix.shape[0]):
    for j in range(metrics_matrix.shape[1]):
        plt.text(j, i, f"{metrics_matrix[i, j]:.2f}", ha="center", va="center", color="black")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'class_metrics.png'))