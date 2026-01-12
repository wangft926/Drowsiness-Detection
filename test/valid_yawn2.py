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


# 每帧进行预测，而不是16个窗口有重复的
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

def get_detector(model_path):
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                           output_face_blendshapes=True,
                                           output_facial_transformation_matrixes=True,
                                           num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)
    return detector


def euclidean_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))


def calculate_ear(landmarks, eye_points):
    # Vertical distances
    v1 = euclidean_distance(landmarks[eye_points[0]], landmarks[eye_points[1]])
    v2 = euclidean_distance(landmarks[eye_points[2]], landmarks[eye_points[3]])
    v3 = euclidean_distance(landmarks[eye_points[4]], landmarks[eye_points[5]])
    # Horizontal distance
    h = euclidean_distance(landmarks[eye_points[6]], landmarks[eye_points[7]])
    # EAR formula
    ear = (v1 + v2 + v3) / (3.0 * h)
    return ear


# Function to calculate Mouth Aspect Ratio (MAR)
def calculate_mar(landmarks, mouth_points):
    # Vertical distances
    v1 = euclidean_distance(landmarks[mouth_points[0]], landmarks[mouth_points[1]])
    v2 = euclidean_distance(landmarks[mouth_points[2]], landmarks[mouth_points[3]])
    v3 = euclidean_distance(landmarks[mouth_points[4]], landmarks[mouth_points[5]])
    # Horizontal distance
    h = euclidean_distance(landmarks[mouth_points[6]], landmarks[mouth_points[7]])
    # MAR formula
    mar = (v1 + v2 + v3) / (3.0 * h)
    return mar


# Define 3D model points of the facial landmarks
model_points = np.array([
    (0.0, 0.0, 0.0),  # Nose tip
    (0.0, -330.0, -65.0),  # Chin
    (-225.0, 170.0, -135.0),  # Left eye left corner
    (225.0, 170.0, -135.0),  # Right eye right corner
    (-150.0, -150.0, -125.0),  # Left mouth corner
    (150.0, -150.0, -125.0)  # Right mouth corner
], dtype=np.float64)


# Function to calculate head pose
def calculate_head_pose(image, landmarks):
    # Image size
    # print(image)
    size = image.shape
    focal_length = size[1]  # Assume focal length equals image width
    center = (size[1] / 2, size[0] / 2)  # Image center

    # Define the camera matrix
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)

    # Assume no lens distortion
    dist_coeffs = np.zeros((4, 1))

    # Solve the PnP problem
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, landmarks, camera_matrix, dist_coeffs
    )

    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    # Construct the 3x4 projection matrix by combining rotation matrix and translation vector
    projection_matrix = np.hstack((rotation_matrix, translation_vector))

    # Decompose the projection matrix to get the Euler angles
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(projection_matrix)

    # Retrieve the angles in degrees
    euler_angles = np.degrees(euler_angles)
    phi = euler_angles[0]  # Left-right rotation
    theta = euler_angles[1]  # Up-down tilt

    return phi, theta


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


def predict_video(video_path, detector, model, window_size=16, overlap=8, norm_file='norm.npy'):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    print(f"总帧数: {frame_count}, 帧率: {fps}, 视频时长: {duration:.2f} 秒")

    r_ear_li = []
    l_ear_li = []
    mar_li = []
    phi_li = []
    theta_li = []
    face_imgs = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        detection_result = detector.detect(image)

        try:
            landmarks = np.array([(lm.x, lm.y) for lm in detection_result.face_landmarks[0]])
        except:
            continue

        r_ear = calculate_ear(landmarks, [160, 144, 159, 145, 158, 153, 33, 133])
        l_ear = calculate_ear(landmarks, [385, 380, 386, 374, 387, 373, 362, 263])
        mar = calculate_mar(landmarks, [81, 178, 13, 14, 311, 402, 78, 308])

        phi, theta = calculate_head_pose(frame, np.array([landmarks[i] for i in [10, 33, 263, 152, 61, 291]]))

        if phi is None or theta is None:
            continue

        face_img = extract_face_from_frame(frame, landmarks)
        face_imgs.append(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB) / 255.0)

        r_ear_li.append(r_ear)
        l_ear_li.append(l_ear)
        mar_li.append(mar)
        # 只保留标量值（和训练一致）
        phi_li.append(phi.item())  # ← 把 shape=(1,) 的 array 转成 float
        theta_li.append(theta.item())

    print("Lengths of features:")
    print("R_EAR:", len(r_ear_li))
    print("L_EAR:", len(l_ear_li))
    print("MAR:", len(mar_li))
    print("Phi:", len(phi_li))
    print("Theta:", len(theta_li))

    if len(r_ear_li) == 0:
        return None, None

    # 后续归一化、滑动窗口等流程不变

    cap.release()

    if len(r_ear_li) == 0:
        return None, None

    # 打印原始特征范围
    print("Raw R_EAR range:", np.min(r_ear_li), np.max(r_ear_li))
    print("Raw MAR range:", np.min(mar_li), np.max(mar_li))

    # 收集所有帧的原始值，然后统一归一化
    all_features = np.vstack([
        np.array(r_ear_li),
        np.array(l_ear_li),
        np.array(mar_li),
        np.array(phi_li),
        np.array(theta_li)
    ])  # shape: (5, T)
    print("All features shape:", all_features.shape)

    # 加载训练阶段保存的 min/max 参数
    norm_params = np.load(norm_file)  # shape: (5, 2)
    print("Norm params shape:", norm_params.shape)
    print("Norm params:\n", norm_params)

    # 对每个特征通道进行统一归一化
    for i in range(5):
        min_val = norm_params[i, 0]
        max_val = norm_params[i, 1]
        all_features[i] = (all_features[i] - min_val) / (max_val - min_val + 1e-8)

    # 【新增】加入裁剪逻辑，强制限制在 [0, 1]
    for i in range(5):
        all_features[i] = np.clip(all_features[i], 0.0, 1.0)

    # 打印归一化后的特征范围（用于调试）
    print("R_EAR 归一化后范围:", np.min(all_features[0]), np.max(all_features[0]))
    print("MAR 归一化后范围:", np.min(all_features[2]), np.max(all_features[2]))

    # 创建滑动窗口（必须与训练时一致）
    windows, face_windows = create_windows(all_features, face_imgs, window_size=window_size, overlap=overlap)
    if len(windows) == 0:
        return None, None

    windows = np.array(windows)
    face_windows = np.array(face_windows)

    # 转换为 Tensor
    windows = torch.tensor(windows, dtype=torch.float32)
    face_windows = torch.tensor(np.transpose(face_windows, (0, 1, 4, 2, 3)), dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    logits_list = []
    with torch.no_grad():
        for i in range(len(windows)):
            geo = windows[i].unsqueeze(0).to(device)
            face = face_windows[i].unsqueeze(0).to(device)
            out = model(geo, face)
            logits_list.append(out.cpu().numpy())

    # Softmax 平均融合
    avg_logits = np.mean(np.vstack(logits_list), axis=0)
    final_pred = int(np.argmax(avg_logits))
    class_probs = F.softmax(torch.tensor(avg_logits), dim=-1).numpy()
    print("Final prediction:", final_pred)
    print("Class probabilities:", class_probs)

    # 返回每个窗口的预测结果和总帧数（可用于可视化分析）
    return [final_pred], frame_count


def create_windows(all_features, all_face_imgs, window_size=16, overlap=8):
    """
    推理专用：将连续的视频特征和图像帧划分为多个窗口样本（支持滑动窗口 + 边界填充）

    参数:
        all_features (np.ndarray): 几何特征矩阵，形状为 (F, T)
        all_face_imgs (list of np.ndarray): 图像帧列表，每个元素是 (H, W, C) 的 numpy 数组
        window_size (int): 每个窗口包含多少帧
        overlap (int): 相邻窗口之间的重叠帧数

    返回:
        windows (list of np.ndarray): 切分后的几何特征窗口列表，每个形状为 (F, window_size)
        face_windows (list of list of np.ndarray): 切分后的图像帧窗口列表，每个子列表含 window_size 张图像
    """
    num_features, num_frames = all_features.shape
    step = window_size - overlap
    windows = []
    face_windows = []

    for start in range(0, num_frames, step):
        end = start + window_size

        if end <= num_frames:
            # 完整窗口
            window = all_features[:, start:end]
            face_window = all_face_imgs[start:end]
        else:
            # 不完整窗口，复制最后一帧补全
            window = all_features[:, start:num_frames]
            missing = window_size - (num_frames - start)
            last_frame = all_features[:, -1:]  # 取最后一帧
            padding = np.repeat(last_frame, missing, axis=1)
            window = np.hstack([window, padding])

            face_window = all_face_imgs[start:num_frames] + [all_face_imgs[-1]] * missing

        assert len(face_window) == window_size, f"Expected {window_size} frames, got {len(face_window)}"

        windows.append(window)
        face_windows.append(face_window)

    return windows, face_windows


def is_fatigue(model_window_outputs, fps, window_size=16, overlap=8):
    """
    model_window_outputs: list of predicted classes per window
    fps: 视频帧率
    window_size: 每个窗口包含的帧数
    overlap: 窗口之间的重叠帧数
    """
    if not model_window_outputs:
        return False, None

    # Step 1: 将窗口预测映射为帧级预测
    frame_predictions = []
    step = window_size - overlap  # 实际移动的帧数

    for i, label in enumerate(model_window_outputs):
        start_idx = i * step
        end_idx = start_idx + window_size
        # 取最后一个窗口的实际长度（防止越界）
        actual_length = min(window_size, len(frame_predictions) + window_size - len(frame_predictions))
        frame_predictions.extend([label] * actual_length)

    # Step 2: 定义帧级变量
    yawn_frames = [i for i, c in enumerate(frame_predictions) if c == 2]
    blink_frames = [i for i, c in enumerate(frame_predictions) if c == 1]

    # Rule 1: 只要出现一次 Yawning（类别2），就判定为疲劳
    if any(c == 2 for c in frame_predictions):
        return True, 1

    # Rule 2: 连续闭眼 ≥ 0.3 秒
    consecutive_blink = 0
    for i in range(len(frame_predictions)):
        if frame_predictions[i] == 1:
            consecutive_blink += 1
            if consecutive_blink / fps >= 0.3:
                return True, 2
        else:
            consecutive_blink = 0

    # Rule 3: PERCLOS > 0.4 （闭眼帧占比）
    perclos = len(blink_frames) / len(frame_predictions)
    if perclos > 0.4:
        return True, 3

    return False, None


# ----------------------------
# 主程序逻辑
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    base_dir = r"G:\A-数据集\YawDD\test"
    model_path = "model-2G.pth"
    mediapipe_model_path = "face_landmarker.task"  # 替换为你自己的路径
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    model = load_model(model_path, device)
    detector = get_detector(mediapipe_model_path)

    results = []

    for root, dirs, files in os.walk(base_dir):
        group_name = os.path.basename(root)
        if not group_name.startswith('group'):
            continue
        print(f"\nProcessing group: {group_name}")
        for file in files:
            # if not file.endswith(".mp4"):
            #     continue
            video_path = os.path.join(root, file)

            # 获取真实标签
            if "Yawning" in file:
                true_label = 2
            elif "Normal" in file:
                true_label = 1
            else:
                true_label = 0

            # 预测
            model_outputs, fps = predict_video(video_path, detector, model)
            if model_outputs is None:
                print(f"跳过 {file}，无法提取特征")
                continue

            # 应用规则
            pred_fatigue, rule_used = is_fatigue(model_outputs, fps)

            results.append({
                "video_name": file,
                # "true_label": true_label, # TODO
                "true_label": 1 if true_label in [2] else 0,
                "model_outputs": model_outputs,
                "pred_label": int(pred_fatigue),
                "rule_used": rule_used if pred_fatigue else np.nan,
                "group": group_name
            })

    print(results)
    df = pd.DataFrame(results)
    df.to_csv("111-fatigue_detection_results.csv", index=False)

    y_true = df["true_label"]
    y_pred = df["pred_label"]

    print("\n=== 混淆矩阵 ===")
    print(confusion_matrix(y_true, y_pred))

    print("\n=== 分类报告 ===")
    print(classification_report(y_true, y_pred, digits=4))

    groups = df.groupby("group")
    print("\n=== 各组准确率 ===")
    for name, group in groups:
        acc = (group["true_label"] == group["pred_label"]).mean()
        print(f"{name}: {acc:.4f}")


if __name__ == "__main__":
    main()
