import cv2
import numpy as np


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
    face_img = face_img / 255.0  # [0,1] 映射
    return face_img
