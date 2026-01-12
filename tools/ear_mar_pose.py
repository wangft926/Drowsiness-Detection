import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
# ----------------------------
# 工具函数定义（完全使用你自己的逻辑）
# ----------------------------

def euclidean_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))


def calculate_ear(landmarks, eye_points):
    v1 = euclidean_distance(landmarks[eye_points[0]], landmarks[eye_points[1]])
    v2 = euclidean_distance(landmarks[eye_points[2]], landmarks[eye_points[3]])
    v3 = euclidean_distance(landmarks[eye_points[4]], landmarks[eye_points[5]])
    h = euclidean_distance(landmarks[eye_points[6]], landmarks[eye_points[7]])
    return (v1 + v2 + v3) / (3.0 * h)


def calculate_mar(landmarks, mouth_points):
    v1 = euclidean_distance(landmarks[mouth_points[0]], landmarks[mouth_points[1]])
    v2 = euclidean_distance(landmarks[mouth_points[2]], landmarks[mouth_points[3]])
    v3 = euclidean_distance(landmarks[mouth_points[4]], landmarks[mouth_points[5]])
    h = euclidean_distance(landmarks[mouth_points[6]], landmarks[mouth_points[7]])
    return (v1 + v2 + v3) / (3.0 * h)


model_points = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye left corner
    (225.0, 170.0, -135.0),      # Right eye right corner
    (-150.0, -150.0, -125.0),    # Left mouth corner
    (150.0, -150.0, -125.0)      # Right mouth corner
], dtype=np.float64)


def calculate_head_pose(image, landmarks):
    size = image.shape
    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    camera_matrix = np.array([[focal_length, 0, center[0]],
                               [0, focal_length, center[1]],
                               [0, 0, 1]], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))
    success, rvec, tvec = cv2.solvePnP(model_points, landmarks, camera_matrix, dist_coeffs)
    if not success:
        return None, None
    rmat, _ = cv2.Rodrigues(rvec)
    _, _, _, _, _, _, angles = cv2.decomposeProjectionMatrix(cv2.hconcat([rmat, tvec]))
    phi = angles[0][0]
    theta = angles[1][0]
    return phi, theta


# ----------------------------
# MediaPipe 初始化
# ----------------------------

def get_detector(model_path):
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                           output_face_blendshapes=True,
                                           output_facial_transformation_matrixes=True,
                                           num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)
    return detector