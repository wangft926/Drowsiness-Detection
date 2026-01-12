import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import matplotlib.pyplot as plt
from config_parser import parse_file
from tools.ear_mar_pose import get_detector
from tools.extract_face_frame import extract_face_from_frame


# 每个视频生成一个npz文件

# -------------------------------
# 工具函数定义
# -------------------------------


# -------------------------------
# 主预处理函数
# -------------------------------
def preprocess_data(config):
    detector = get_detector(config['mediapipe_model_path'])
    video_root = config['train_dataset_path']
    label_root = config['label_csv_path']
    window_size = config['window_size']
    output_dir = config['output_dir']

    os.makedirs(output_dir, exist_ok=True)

    # 存储全局 mean & std
    all_features_list = []

    # 获取所有视频文件（不考虑子目录）
    video_files = [f for f in os.listdir(video_root) if f.endswith(('.mp4', '.avi', '.mov'))]

    # 🟢 添加全局进度条
    pbar = tqdm(total=len(video_files), desc="📊 Overall Progress", position=0)

    for video_file in video_files:
        video_name = os.path.splitext(video_file)[0]
        video_path = os.path.join(video_root, video_file)
        csv_path = os.path.join(label_root, f"{video_name}.csv")

        # 检查 CSV 是否存在
        if not os.path.exists(csv_path):
            print(f"❌ Missing label file: {csv_path}")
            pbar.update(1)  # 🟢 更新进度条
            continue

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video: {video_path}")
            pbar.update(1)  # 🟢 更新进度条
            continue

        df = pd.read_csv(csv_path)
        features = []
        face_imgs = []
        labels = []

        # 🟢 新增 valid_indices 来记录哪些帧被成功检测到了
        valid_indices = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            try:
                row = df.iloc[frame_idx]
            except IndexError:
                continue
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = detector.detect(image)

            try:
                landmarks = np.array([(lm.x, lm.y) for lm in detection_result.face_landmarks[0]])
            except Exception as e:
                print(f"No landmark detected at frame {frame_idx}: {e}")
                continue

            r_ear = row['right_ear']
            l_ear = row['left_ear']
            mar = row['mar']
            phi = row['phi']
            theta = row['theta']
            label = row['label']

            face_img = extract_face_from_frame(frame, landmarks)
            # cv2.imwrite("face.png", (face_img * 255).astype(np.uint8))  # 恢复到 [0,255]
            face_imgs.append(face_img)
            features.append([r_ear, l_ear, mar, phi, theta])
            labels.append(label)
            valid_indices.append(frame_idx)  # 记录有效帧索引
        cap.release()

        if len(features) < window_size:
            print(f"Video too short: {video_name}, length={len(features)}")
            pbar.update(1)  # 🟢 更新进度条
            continue

        features = np.array(features)  # shape: (T, 5)
        face_imgs = np.array(face_imgs)  # shape: (T, H, W, C)
        labels = np.array(labels)  # shape: (T, )
        # ✅ 确保三者长度一致
        assert len(features) == len(face_imgs) == len(labels), \
            f"Length mismatch: features={len(features)}, face_imgs={len(face_imgs)}, labels={len(labels)}"
        print("Features shape:", features.shape)
        print("Labels shape:", labels.shape)
        print("Face images shape:", face_imgs.shape)
        np.save(os.path.join(output_dir, f"valid_indices_{video_name}.npy"), np.array(valid_indices))

        # 收集用于计算 mean & std 的数据
        all_features_list.append(features.copy())

        # 滑动窗口切片
        geo_windows = []
        face_windows = []
        label_windows = []

        for i in range(window_size, len(features) + 1):
            start = i - window_size
            end = i
            window_feat = features[start:end]  # shape: (W, 5)
            window_face = face_imgs[start:end]  # shape: (W, H, W, C)
            window_label = labels[end - 1]  # 最后一帧的 label

            geo_windows.append(window_feat)
            face_windows.append(window_face)
            label_windows.append(window_label)

        geo_windows = np.array(geo_windows)  # shape: (N, W, 5)
        face_windows = np.array(face_windows)  # shape: (N, W, H, W, C)
        label_windows = np.array(label_windows)  # shape: (N, )

        # 保存为 .npz 文件（推荐方式）
        output_path = os.path.join(output_dir, f"{video_name}.npz")
        print(f"Saving {video_name}.npz")
        print("geo_windows.shape:", geo_windows.shape)
        print("face_windows.shape:", face_windows.shape)
        np.savez_compressed(output_path,
                            get_features=geo_windows,
                            face_images=face_windows,
                            labels=label_windows)
        pbar.update(1)  # 🟢 更新进度条：每处理完一个视频就 +1
    pbar.close()  # 🟢 关闭进度条

    # 计算全局 mean & std
    all_features = np.vstack(all_features_list)
    mean = np.mean(all_features, axis=0)
    std = np.std(all_features, axis=0)

    # 保存 mean & std 到 .npz 文件
    stats_path = os.path.join(output_dir, "feature_stats.npz")
    np.savez_compressed(stats_path, mean=mean, std=std)

    print("✅ Preprocessing complete.")
    print(f"📊 Mean: {mean}, Std: {std}")
    print(f"💾 Feature stats saved to: {stats_path}")

    return mean, std


config_file = parse_file('config.ini')
mean, std = preprocess_data(config_file)
