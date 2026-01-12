import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from tools.ear_mar_pose import calculate_ear, calculate_mar, calculate_head_pose, get_detector


# ----------------------------
# 主程序逻辑
# ----------------------------

def process_video(video_path, detector, label_dir):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    label_file = os.path.join(label_dir, f"{video_name}.csv")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return

    results = []
    frame_idx = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = detector.detect(mp_image)

        if len(detection_result.face_landmarks) == 0:
            # 没有人脸检测结果，标记为正常，并设置 valid=0
            results.append({
                'frame_num': frame_idx,
                'left_ear': np.nan,
                'right_ear': np.nan,
                'ear': np.nan,
                'mar': np.nan,
                'phi': np.nan,
                'theta': np.nan,
                'label': 1,
                'valid': 0
            })
            frame_idx += 1
            continue

        landmarks = np.array([(fl.x, fl.y) for fl in detection_result.face_landmarks[0]])

        try:
            # 计算左右眼 EAR
            right_ear = calculate_ear(landmarks, [160, 144, 159, 145, 158, 153, 33, 133])
            left_ear = calculate_ear(landmarks, [385, 380, 386, 374, 387, 373, 362, 263])
            ear = (right_ear + left_ear) / 2.0

            # 计算 MAR
            mar = calculate_mar(landmarks, [81, 178, 13, 14, 311, 402, 78, 308])

            # Head pose
            landmarks_2d = np.array([landmarks[i] for i in [10, 33, 263, 152, 61, 291]])
            landmarks_2d[:, 0] *= w
            landmarks_2d[:, 1] *= h
            phi, theta = calculate_head_pose(frame, landmarks_2d)

            # 判断标签（MAR > 0.7 → 2；EAR < 0.2 → 0）
            if mar > 0.75:
                label = 2
            # elif right_ear < 0.2 or left_ear < 0.2:
            elif right_ear < 0.2 or left_ear < 0.2:
                label = 0
            else:
                label = 1

            valid = 1  # 成功检测和计算

        except Exception as e:
            print(f"Error calculating features on frame {frame_idx}: {e}")
            left_ear = np.nan
            right_ear = np.nan
            ear = np.nan
            mar = np.nan
            phi = np.nan
            theta = np.nan
            label = 1
            valid = 0  # 异常情况，默认正常

        results.append({
            'frame_num': frame_idx,
            'left_ear': left_ear,
            'right_ear': right_ear,
            'ear': ear,
            'mar': mar,
            'phi': phi,
            'theta': theta,
            'label': label,
            'valid': valid
        })

        frame_idx += 1

    cap.release()

    df = pd.DataFrame(results)
    df.to_csv(label_file, index=False)
    print(f"Saved labels to {label_file}")


# ----------------------------
# 启动主程序
# ----------------------------

if __name__ == '__main__':
    dataset_dir = r'../YawDD非36个的外部验证数据集'  # 视频目录
    label_dir = r'../YawDD非36个的外部验证数据集_label'  # 输出 CSV 目录
    #label_dir = '../dataset_label'  # 输出 CSV 目录
    model_path = '../face_landmarker.task'  # MediaPipe模型路径

    os.makedirs(label_dir, exist_ok=True)

    detector = get_detector(model_path)

    video_files = [f for f in os.listdir(dataset_dir)]
    for video_file in tqdm(video_files, desc="Processing videos"):
        video_path = os.path.join(dataset_dir, video_file)
        process_video(video_path, detector, label_dir)
