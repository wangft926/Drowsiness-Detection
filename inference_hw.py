import time

import cv2
import mediapipe as mp
import numpy as np
import serial  # For Arduino communication
import torch
import torch.nn as nn

# Setup Arduino serial communication
arduino = serial.Serial('COM3', 9600)  # Replace 'COM_PORT' with your Arduino port (e.g., 'COM3' or '/dev/ttyACM0')
time.sleep(2)  # Wait for Arduino to initialize

from config_parser import parse_file
from dataset import get_detector
from draw_landmarks import *
from model import CNN_LSTM

# Load configuration file and model
config_file = parse_file('config.ini')
model = CNN_LSTM(config_file)
model.load_state_dict(torch.load(config_file['model_path'], weights_only=True))
model.eval()
min_max_array = np.load(config_file['normalize_data_file'])
print(min_max_array)
frame_buffer = []

# Capture video from the webcam
cap = cv2.VideoCapture(0)  # Change 0 if using a different camera source

fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video = cv2.VideoWriter(config_file['inference_video_out'], fourcc, cap.get(cv2.CAP_PROP_FPS), (frame_width, frame_height))

r_ear_li = []
l_ear_li = []
mar_li = []
phi_li = []
theta_li = []
detector = get_detector(config_file['mediapipe_model_path'])

cnt = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Process the frame
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    detection_result = detector.detect(image)

    try:
        landmarks = np.array([(face_landmarks.x, face_landmarks.y) for face_landmarks in detection_result.face_landmarks[0]])
    except:
        print('No landmark')
        continue

    r_ear = calculate_ear(landmarks, [160, 144, 159, 145, 158, 153, 33, 133])
    l_ear = calculate_ear(landmarks, [385, 380, 386, 374, 387, 373, 362, 263])
    mar = calculate_mar(landmarks, [81, 178, 13, 14, 311, 402, 78, 308])
    phi, theta = calculate_head_pose(frame, np.array([landmarks[i] for i in [10, 33, 263, 152, 61, 291]]))

    r_ear_li.append((r_ear - min_max_array[0][0]) / (min_max_array[0][1] - min_max_array[0][0]))
    l_ear_li.append((l_ear - min_max_array[1][0]) / (min_max_array[1][1] - min_max_array[1][0]))
    mar_li.append((mar - min_max_array[2][0]) / (min_max_array[2][1] - min_max_array[2][0]))
    phi_li.append((phi[0] - min_max_array[3][0]) / (min_max_array[3][1] - min_max_array[3][0]))
    theta_li.append((theta[0] - min_max_array[4][0]) / (min_max_array[4][1] - min_max_array[4][0]))
    all_features = np.vstack([r_ear_li, l_ear_li, mar_li, phi_li, theta_li])

    frame_buffer.append(all_features[:, cnt])
    
    if len(frame_buffer) > 16:
        frame_buffer.pop(0)
    
    if len(frame_buffer) == 16:
        fr = np.array(frame_buffer)
        fr = np.transpose(fr, (1, 0))
        input_data = torch.tensor(fr, dtype=torch.float32).unsqueeze(axis=0)

        # Make prediction
        prediction = model(torch.tensor(input_data))
        
        # Output the result
        if torch.argmax(nn.Softmax(dim=1)(prediction), 1) == 0:
            print("Driver is Attentive.")
            cv2.putText(frame, "Normal", (80, 80), 2, 2, (0, 255, 0))
            arduino.write(b'0')  # No drowsiness, send '0' to Arduino
        elif torch.argmax(nn.Softmax(dim=1)(prediction), 1) == 1:
            print("Alert: Drowsiness Detected!")
            cv2.putText(frame, 'Drowsiness Detected', (100, 100), 2, 2, (0, 0, 255))
            arduino.write(b'1')  # Drowsiness detected, send '1' to Arduino to activate buzzer
        cnt += 1

    # Display the video feed (optional)
    cv2.imshow('Live Video', frame)

    # Exit condition
    if cv2.waitKey(1) == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
arduino.close()  # Close Arduino connection
