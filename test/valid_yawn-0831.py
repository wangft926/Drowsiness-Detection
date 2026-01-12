import os
import cv2
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from torch.nn.init import xavier_uniform_
import torch.nn as nn
import torch.nn.functional as F
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tqdm import tqdm
from torchvision.models import mobilenet_v3_small

from tools.ear_mar_pose import calculate_ear, calculate_mar, calculate_head_pose, get_detector


# 每当有第 n 帧时（从第 16 帧开始），就使用 [n-15, n] 这 16 帧数据作为输入进行一次预测，输出当前帧的预测结果。
# ----------------------------
# 模型定义（与训练时一致）
# ----------------------------

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=2):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_k = nn.LayerNorm(embed_dim)
        self.norm_v = nn.LayerNorm(embed_dim)
        self.proj_out = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, query, key, value):
        q = self.norm_q(query)
        k = self.norm_k(key)
        v = self.norm_v(value)
        attn_output, _ = self.attn(q, k, v)
        fused = torch.mean(attn_output, dim=1)
        return self.proj_out(fused)


class FrameFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = mobilenet_v3_small(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        self.proj = nn.Linear(576, 128)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        features = self.feature_extractor(x)
        features = features.view(B * T, -1)
        features = self.proj(features)
        features = features.view(B, T, -1)
        return features


class GatedFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        gate_input = torch.cat([x1, x2], dim=-1)
        gate = self.gate(gate_input)
        return gate * x1 + (1 - gate) * x2


class DualInputModel(nn.Module):
    def __init__(self, config):
        super(DualInputModel, self).__init__()

        # 几何特征分支
        self.gru_geo = nn.GRU(input_size=config['num_features'], hidden_size=128,
                              bidirectional=True, batch_first=True)
        self.norm_geo = nn.LayerNorm(256)
        self.proj_geo = nn.Linear(256, 128)

        # 图像特征分支
        self.cnn_img = FrameFeatureExtractor()
        self.gru_img = nn.GRU(input_size=128, hidden_size=128,
                              bidirectional=True, batch_first=True)
        self.norm_img = nn.LayerNorm(256)
        self.proj_img = nn.Linear(256, 128)

        # 注意力融合
        self.learnable_queries = nn.Parameter(torch.randn(1, 1, 128))
        self.cross_attn_geo_to_img = CrossAttention(embed_dim=128)
        self.cross_attn_img_to_geo = CrossAttention(embed_dim=128)
        self.gated_fusion = GatedFusion(dim=128)

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.BatchNorm1d(64), nn.Dropout(0.5),
            nn.Linear(64, config['num_classes'])
        )

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, geo_input, face_input):
        B, T = face_input.shape[0], face_input.shape[1]

        # 处理几何特征
        geo_input = geo_input.transpose(1, 2)
        gru_geo_out, _ = self.gru_geo(geo_input)
        gru_geo_out = self.norm_geo(gru_geo_out)
        geo_lstm = self.proj_geo(gru_geo_out)

        # 处理图像特征
        face_out = self.cnn_img(face_input)
        gru_img_out, _ = self.gru_img(face_out)
        gru_img_out = self.norm_img(gru_img_out)
        face_out = self.proj_img(gru_img_out)

        # 注意力交互
        queries = self.learnable_queries.expand(B, -1, -1)
        combined_geo = self.cross_attn_geo_to_img(query=queries, key=face_out, value=face_out)
        combined_img = self.cross_attn_img_to_geo(query=queries, key=geo_lstm, value=geo_lstm)

        # 融合
        final = self.gated_fusion(combined_geo, combined_img)
        out = self.classifier(final)
        return out


# ----------------------------
# 工具函数定义
# ----------------------------


def extract_face_from_frame(frame, landmarks, padding=0.2):
    h, w, _ = frame.shape
    landmarks_abs = np.copy(landmarks)
    landmarks_abs[:, 0] *= w
    landmarks_abs[:, 1] *= h
    x_min, y_min = np.min(landmarks_abs, axis=0).astype(int)
    x_max, y_max = np.max(landmarks_abs, axis=0).astype(int)

    pad_x = int((x_max - x_min) * padding)
    pad_y = int((y_max - y_min) * padding)

    x_min = max(x_min - pad_x, 0)
    y_min = max(y_min - pad_y, 0)
    x_max = min(x_max + pad_x, w)
    y_max = min(y_max + pad_y, h)

    face_img = frame[y_min:y_max, x_min:x_max]
    face_img = cv2.resize(face_img, (128, 128))  # 固定尺寸
    return face_img


# ----------------------------
# 加载模型 & 推理函数
# ----------------------------

def load_model(model_path, device):
    config = {
        'num_features': 5,
        'num_classes': 3
    }
    model = DualInputModel(config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


import numpy as np
import cv2
import torch
import mediapipe as mp
import torch.nn.functional as F


def predict_video(video_path, detector, model, window_size=10, norm_file='norm.npy', show_progress=False):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    print(f"视频名称：{video_path}, 总帧数: {frame_count}, 帧率: {fps}, 视频时长: {duration:.2f} 秒")

    r_ear_li = []
    l_ear_li = []
    mar_li = []
    phi_li = []
    theta_li = []
    face_imgs = []
    predictions = [None] * frame_count  # 初始化预测列表

    # 不显示进度条，只在开始和结束时打印信息
    processed_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        idx = len(r_ear_li)  # 当前帧索引

        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        detection_result = detector.detect(image)

        try:
            landmarks = np.array([(lm.x, lm.y) for lm in detection_result.face_landmarks[0]])
        except:
            # 如果检测不到人脸，保留 None 预测
            processed_frames += 1
            continue

        r_ear = calculate_ear(landmarks, [160, 144, 159, 145, 158, 153, 33, 133])
        l_ear = calculate_ear(landmarks, [385, 380, 386, 374, 387, 373, 362, 263])
        mar = calculate_mar(landmarks, [81, 178, 13, 14, 311, 402, 78, 308])

        phi, theta = calculate_head_pose(frame, np.array([landmarks[i] for i in [10, 33, 263, 152, 61, 291]]))

        if phi is None or theta is None:
            processed_frames += 1
            continue

        face_img = extract_face_from_frame(frame, landmarks)
        face_imgs.append(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB) / 255.0)

        r_ear_li.append(r_ear)
        l_ear_li.append(l_ear)
        mar_li.append(mar)
        phi_li.append(phi.item())
        theta_li.append(theta.item())

        # 只要收集到足够的帧数，就开始预测
        if len(r_ear_li) >= window_size:
            # 提取最近 window_size 帧特征
            start_idx = len(r_ear_li) - window_size
            current_r_ear = r_ear_li[start_idx:start_idx + window_size]
            current_l_ear = l_ear_li[start_idx:start_idx + window_size]
            current_mar = mar_li[start_idx:start_idx + window_size]
            current_phi = phi_li[start_idx:start_idx + window_size]
            current_theta = theta_li[start_idx:start_idx + window_size]
            current_face = face_imgs[start_idx:start_idx + window_size]

            # 构造单个窗口
            all_features = np.vstack([
                np.array(current_r_ear),
                np.array(current_l_ear),
                np.array(current_mar),
                np.array(current_phi),
                np.array(current_theta)
            ])  # shape: (5, window_size)

            # 加载归一化参数
            norm_params = np.load(norm_file)  # shape: (5, 2)

            # 归一化  TODO 0713
            # ✅ 使用 Z-Score 归一化（和训练时保持一致）
            means = norm_params[0]  # 第一行是均值
            stds = norm_params[1]  # 第二行是标准差
            for i in range(5):
                mean = means[i]
                std = stds[i]
                all_features[i] = (all_features[i] - mean) / (std + 1e-8)  # 防止除以零
                all_features[i] = np.clip(all_features[i], -5.0, 5.0)  # 可选：防止异常值

            # 图像窗口处理
            face_window = np.array(current_face)
            face_window = torch.tensor(np.transpose(face_window, (0, 3, 1, 2)), dtype=torch.float32).unsqueeze(0)

            # 转换为 Tensor
            geo_window = torch.tensor(all_features, dtype=torch.float32).unsqueeze(0)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            model.eval()

            with torch.no_grad():
                out = model(geo_window.to(device), face_window.to(device))
                logits = out.cpu().numpy()[0]
                pred = int(np.argmax(logits))
                prob = F.softmax(torch.tensor(logits), dim=-1).numpy()

            predictions[idx] = pred
            # print(f"Frame {idx + 1}: Predicted class {pred}, Probabilities: {prob}")
        else:
            predictions[idx] = None  # 前 window_size - 1 帧不预测

        processed_frames += 1

    cap.release()
    print(f"完成处理：{video_path} ({processed_frames}/{frame_count} 帧)")

    # 返回每个帧的预测结果和帧率
    return predictions, fps


def is_fatigue(model_window_outputs, fps):
    """
    model_window_outputs: list of predicted classes per frame (length = total_frames)
    fps: 视频帧率
    """
    if not model_window_outputs or all(x is None for x in model_window_outputs):
        return False, None

    # 去掉前面的 None
    frame_predictions = [x for x in model_window_outputs if x is not None]

    # Rule 1: 连续出现3次Yawning（类别2），就判定为疲劳
    consecutive_Yawning = 0
    for i in range(len(frame_predictions)):
        if frame_predictions[i] == 2:  # 打哈欠为2
            consecutive_Yawning += 1
            if consecutive_Yawning >= 10:
                print("consecutive_Yawning" + str(consecutive_Yawning))
                return "Yawning", 1
        else:
            consecutive_Yawning = 0
    # if any(c == 2 for c in frame_predictions):
    #     return "Yawning", 1
    # Rule 1: 连续闭眼10帧，Yawning（类别2），就判定为疲劳
    # 或者判断十帧有6帧
    consecutive_CloseEye = 0
    for i in range(len(frame_predictions)):
        if frame_predictions[i] == 0:  # 闭眼为0
            consecutive_CloseEye += 1
            if consecutive_CloseEye >= 20:
                print("consecutive_CloseEye" + str(consecutive_CloseEye))
                return "Yawning", 2
        else:
            consecutive_CloseEye = 0

    # # Rule 2: 连续闭眼 ≥ 0.3 秒
    # consecutive_blink = 0
    # for i in range(len(frame_predictions)):
    #     if frame_predictions[i] == 0:
    #         consecutive_blink += 1
    #         if consecutive_blink / fps >= 0.35:
    #             print("consecutive_blink" + str(consecutive_blink / fps))
    #             return "Yawning", 3
    #     else:
    #         consecutive_blink = 0

    # Rule 3: PERCLOS >= 0.4 （闭眼帧占比）
    blink_count = sum(1 for c in frame_predictions if c == 0)
    perclos = blink_count / len(frame_predictions)
    if perclos >= 0.4:
        print("perclos" + str(perclos))
        return "Yawning", 4

    # 返回正常状态
    return "Normal", None


# ----------------------------
# 主程序逻辑
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    base_dir = r"G:\A-数据集\YawDD\test"
    model_path = "model.pth"
    mediapipe_model_path = "face_landmarker.task"  # 替换为你自己的路径
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    model = load_model(model_path, device)
    detector = get_detector(mediapipe_model_path)

    results = []

    # 收集所有文件
    all_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            all_files.append((root, file))

    # 文件处理进度条
    file_pbar = tqdm(total=len(all_files), desc="处理视频文件", unit="个")

    for idx, (root, file) in enumerate(all_files):
        video_path = os.path.join(root, file)

        # 获取真实标签
        if "Yawning" in file:
            true_label = "Yawning"  # 疲劳
        elif "Normal" in file:
            true_label = "Normal"  # 正常
        else:
            true_label = "Normal"  # 正常

        # 更新进度条描述
        file_pbar.set_description(f"正在处理: {file}")
        file_pbar.refresh()

        # 预测（不显示帧级别进度条）
        model_outputs, fps = predict_video(video_path, detector, model, show_progress=False)
        if model_outputs is None:
            print(f"跳过 {file}，无法提取特征")
            file_pbar.update(1)
            continue

        # 应用规则
        pred_fatigue, rule_used = is_fatigue(model_outputs, fps)

        results.append({
            "video_name": file,
            "true_label": true_label,
            "pred_label": pred_fatigue,
            "rule_used": rule_used
        })

        # 更新文件进度条
        file_pbar.update(1)

    file_pbar.close()

    print(results)
    df = pd.DataFrame(results)
    df.to_csv("fatigue_detection_results.csv", index=False)

    # 后续代码保持不变...
    y_true = df["true_label"]
    y_pred = df["pred_label"]

    print("\n=== 混淆矩阵 ===")
    print(confusion_matrix(y_true, y_pred))

    print("\n=== 分类报告 ===")
    print(classification_report(y_true, y_pred, digits=4))

    # 计算Yawning预测的准确率
    yawning_true_mask = (df["true_label"] == "Yawning")
    yawning_true_count = yawning_true_mask.sum()

    # 初始化变量以避免未定义错误
    yawning_correct = 0
    yawning_accuracy = 0.0

    if yawning_true_count > 0:
        yawning_correct = ((df["true_label"] == "Yawning") & (df["pred_label"] == "Yawning")).sum()
        yawning_accuracy = yawning_correct / yawning_true_count
        print(f"\n=== Yawning预测准确率 ===")
        print(f"真实为Yawning的样本数: {yawning_true_count}")
        print(f"正确预测为Yawning的样本数: {yawning_correct}")
        print(f"Yawning预测准确率: {yawning_accuracy:.4f} ({yawning_correct}/{yawning_true_count})")
    else:
        print("\n=== Yawning预测准确率 ===")
        print("数据集中没有Yawning样本，无法计算Yawning预测准确率")

    # 计算总体准确率
    total_correct = (y_true == y_pred).sum()
    total_samples = len(y_true)
    overall_accuracy = total_correct / total_samples
    print(f"\n=== 总体准确率 ===")
    print(f"总体准确率: {overall_accuracy:.4f} ({total_correct}/{total_samples})")

    # 将关键统计结果写入CSV文件
    stats_results = [{
        "真实为Yawning的样本数": yawning_true_count,
        "正确预测为Yawning的样本数": yawning_correct if yawning_true_count > 0 else 0,
        "Yawning预测准确率": f"{yawning_accuracy:.4f}" if yawning_true_count > 0 else "N/A",
        "总体准确率": f"{overall_accuracy:.4f}",
        "总样本数": total_samples,
        "正确预测数": total_correct
    }]

    stats_df = pd.DataFrame(stats_results)
    stats_df.to_csv("detection_statistics.csv", index=False)


if __name__ == "__main__":
    main()
