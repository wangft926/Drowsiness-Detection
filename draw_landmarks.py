from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def draw_landmarks_on_image(rgb_image, detection_result):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected faces to visualize.
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        # Draw the face landmarks.
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for idx, landmark in
            enumerate(face_landmarks)
        ])

        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_tesselation_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_contours_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_iris_connections_style())

    return annotated_image


def euclidean_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))


# Function to calculate Eye Aspect Ratio (EAR)
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


def annotate_points(landmark_li, idx_li, annotated_image, h, w, color):
    for idx in idx_li:
        # Get the x, y coordinates of the landmark
        x = int(landmark_li[idx][0] * w)
        y = int(landmark_li[idx][1] * h)

        # Draw the landmark as a small circle on the frame
        cv2.circle(annotated_image, (x, y), 5, color, -1)
    return annotated_image
