import time  # Import the time module

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn

from config_parser import parse_file
from dataset import get_detector
from draw_landmarks import *
from model import CNN_LSTM

config_file = parse_file('config.ini')
model = CNN_LSTM(config_file)
model.load_state_dict(torch.load(config_file['model_path'], weights_only=True))
model.eval()
min_max_array = np.load(config_file['normalize_data_file'])
print(min_max_array)
frame_buffer = []

# Capture video from the webcam (or any other source)
cap = cv2.VideoCapture(0)  # Change 0 to the index of your camera or video source

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

prediction_list = []
cnt = 0
last_prediction = None  # Store the last prediction
last_detection_time = None  # Store the last time a state was detected

while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    # Load the input image.
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    # Detect face landmarks from the input image.
    detection_result = detector.detect(image)

    try:
        landmarks = np.array([(face_landmarks.x, face_landmarks.y) for face_landmarks in detection_result.face_landmarks[0]])
    except:
        print('No landmark')
        continue

    r_ear = calculate_ear(landmarks, [160, 144, 159, 145, 158, 153, 33, 133])
    l_ear = calculate_ear(landmarks, [385, 380, 386, 374, 387, 373, 362, 263])
    mar = calculate_mar(landmarks, [81, 178, 13, 14, 311, 402, 78, 308])
    phi, theta = calculate_head_pose(frame, np.array([landmarks[i] for i in [10,33,263,152,61,291]]))

    r_ear_li.append((r_ear - min_max_array[0][0]) / (min_max_array[0][1] - min_max_array[0][0]))
    l_ear_li.append((l_ear - min_max_array[1][0]) / (min_max_array[1][1] - min_max_array[1][0]))
    mar_li.append((mar - min_max_array[2][0]) / (min_max_array[2][1] - min_max_array[2][0]))
    phi_li.append((phi[0] - min_max_array[3][0]) / (min_max_array[3][1] - min_max_array[3][0]))
    theta_li.append((theta[0] - min_max_array[4][0]) / (min_max_array[4][1] - min_max_array[4][0]))
    all_features = np.vstack([r_ear_li,l_ear_li,mar_li,phi_li,theta_li])

    frame_buffer.append(all_features[:,cnt])
    
    # Keep the buffer to 16 frames (sliding window)
    if len(frame_buffer) > 16:
        frame_buffer.pop(0)

    # Step 3: If buffer has 16 frames, process them through the CNN-LSTM model
    if len(frame_buffer) == 16:
        fr = np.array(frame_buffer)
        fr = np.transpose(fr , (1,0))
        input_data = torch.tensor(fr, dtype=torch.float32).unsqueeze(axis=0)

        # Make prediction
        prediction = model(input_data)

        # Determine the current prediction
        current_prediction = torch.argmax(nn.Softmax(dim=1)(prediction), 1).item()  # Get the predicted class as an integer

        # Update last_detection_time if the prediction changes
        if current_prediction != last_prediction:
            last_detection_time = time.time()  # Reset the timer
            last_prediction = current_prediction

        # Check if we need to delay the output
        if last_detection_time is not None and time.time() - last_detection_time < 5:  # Check if 5 seconds have passed
            if current_prediction == 0:
                display_text = "Driver is Attentive."
                prediction_list.append('Normal')
            elif current_prediction == 1:
                display_text = "Alert: Drowsiness Detected!"
                prediction_list.append('Yawning')
            else:
                display_text = "Alert: Drowsiness Detected!"
                prediction_list.append('Eye close')
        else:
            # Reset prediction if more than 5 seconds have passed
            last_prediction = None
            display_text = "Normal"
            prediction_list.append('Normal')

        cnt += 1

    # Display the prediction only if it has stabilized
    if len(prediction_list) > 5:
        if all(prediction == prediction_list[-1] for prediction in prediction_list[-5:]):
            cv2.putText(frame, display_text, (100, 100), 2, 2, (0, 0, 255))
        prediction_list.pop(0)

    # Display the video feed (optional)
    cv2.imshow('Live Video', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

    video.write(frame)

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
