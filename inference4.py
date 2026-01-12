import cv2
from config_parser import parse_file
import torch
from model import CNN_LSTM
from dataset import get_detector
import mediapipe as mp
import numpy as np
from draw_landmarks import *
import torch.nn as nn

config_file = parse_file('config.ini')
model = CNN_LSTM(config_file)
model.load_state_dict(torch.load(config_file['model_path'], weights_only=True))
model.eval()
min_max_array = np.load(config_file['normalize_data_file'])
print(min_max_array)
frame_buffer = []

# Capture video from the webcam (or any other source)
# cap = cv2.VideoCapture('/home/kitty/Downloads/Yawning(1)/Yawning/Yawning/Yawning_95.avi')  # Change 0 to the index of your camera or video source
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
while True:
    ret, frame = cap.read()
    
    if not ret:
        break


    # STEP 3: Load the input image.
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    # STEP 4: Detect face landmarks from the input image.
    detection_result = detector.detect(image)
    # STEP 5: Process the detection result. In this case, visualize it.
    #annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
    # cv2.imshow('img',cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    try:
        landmarks = np.array([(face_landmarks.x, face_landmarks.y) for face_landmarks in detection_result.face_landmarks[0]])
    except:
        print('No landmark')
        # cv2.imshow('frame',frame)
        # cv2.waitKey(0)
        continue
    r_ear = calculate_ear(landmarks, [160, 144, 159, 145, 158, 153, 33, 133])
    # Left Eye Aspect Ratio (L-EAR) calculation
    l_ear = calculate_ear(landmarks, [385, 380, 386, 374, 387, 373, 362, 263])
     # Mouth Aspect Ratio (MAR) calculation
    mar = calculate_mar(landmarks, [81, 178, 13, 14, 311, 402, 78, 308])
    # Print the values
    phi, theta = calculate_head_pose(frame, np.array([landmarks[i] for i in [10,33,263,152,61,291]]))

    r_ear_li.append((r_ear - min_max_array[0][0]) / (min_max_array[0][1] - min_max_array[0][0]))
    l_ear_li.append((l_ear - min_max_array[1][0]) / (min_max_array[1][1] - min_max_array[1][0]))
    mar_li.append((mar - min_max_array[2][0]) / (min_max_array[2][1] - min_max_array[2][0]))
    phi_li.append((phi[0] - min_max_array[3][0]) / (min_max_array[3][1] - min_max_array[3][0]))
    theta_li.append((theta[0] - min_max_array[4][0]) / (min_max_array[4][1] - min_max_array[4][0]))
    all_features = np.vstack([r_ear_li,l_ear_li,mar_li,phi_li,theta_li])

    frame_buffer.append(all_features[:,cnt])
    
    # # Keep the buffer to 16 frames (sliding window)
    if len(frame_buffer) > 16:
        frame_buffer.pop(0)
    
    # # Step 3: If buffer has 16 frames, process them through the CNN-LSTM model
    if len(frame_buffer) == 16:
        fr = np.array(frame_buffer)
        fr = np.transpose(fr , (1,0))
        print(fr.shape)
        # input_data = np.array(frame_buffer).reshape(1, 1)  # Reshape to (1, 16 frames, 6 features)
        input_data = torch.tensor(fr, dtype=torch.float32).unsqueeze(axis=0)
        print(input_data.shape)
        # Make prediction
        prediction = model(torch.tensor(input_data))
        
        # Step 4: Output the result (for example, drowsiness or alertness detection)
        if torch.argmax(nn.Softmax(dim=1)(prediction),1) == 0:
            print("Driver is Attentive.")
            # cv2.putText(frame,"Normal", (100,100), 2, 2, (0,255,0))
            prediction_list.append('Normal')
        elif torch.argmax(nn.Softmax(dim=1)(prediction),1) == 1:
            print("Alert: Drowsiness Detected!")
            # cv2.putText(frame,'Yawning',(100,100), 2, 2, (0,0,255))
            prediction_list.append('Yawning')
        else:
            print("Alert: Drowsiness Detected!")
            # cv2.putText(frame,'Eye close',(100,100), 3, 2, (0,0,255))
            prediction_list.append('Eye close')

        cnt+=1

    if len(prediction_list) > 2:
        if (prediction_list[-1] == prediction_list[-2]) and (prediction_list[-1] == prediction_list[-3]):
            cv2.putText(frame,prediction_list[-1],(100,100), 2, 2, (0,0,255))
        prediction_list.pop(0)


    # Display the video feed (optional)
    cv2.imshow('Live Video', frame)
    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break
    # cv2.imwrite('output/frame_'+str(cnt)+'.png',frame)
    video.write(frame)

# Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
