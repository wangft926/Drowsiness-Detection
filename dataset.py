import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import cv2
from draw_landmarks import *
import numpy as np
from tqdm import tqdm


def get_detector(model_path):
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                           output_face_blendshapes=True,
                                           output_facial_transformation_matrixes=True,
                                           num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)
    return detector


def create_windows(all_features, all_face_imgs, label, class_idx, window_size=16, overlap=8):
    num_features, num_frames = all_features.shape
    step = window_size - overlap
    windows = []
    face_windows = []
    labels = []
    # Sliding window approach
    for start in range(0, num_frames, step):
        end = start + window_size

        if end <= num_frames:
            # If the window is fully within the number of frames, take it as is
            window = all_features[:, start:end]
            face_window = all_face_imgs[start:end]
        else:
            # If we reach the end and the window is incomplete, pad by duplicating the last frame
            window = all_features[:, start:num_frames]
            # 不足 window_size 的情况
            missing = window_size - (num_frames - start)
            last_frame = all_features[:, -1:]  # Take the last frame

            # Duplicate the last frame to fill the remaining space
            padding = np.repeat(last_frame, missing, axis=1)
            window = np.hstack([window, padding])

            face_window = all_face_imgs[start:num_frames] + [all_face_imgs[-1]] * missing
        # 断言检查长度
        assert len(face_window) == window_size, f"Expected {window_size} frames, got {len(face_window)}"
        windows.append(window)
        face_windows.append(face_window)
        labels.append(class_idx[label])

    return windows, face_windows, labels


def preprocess_data(config):
    detector = get_detector(config['mediapipe_model_path'])
    directory = config['train_dataset_path']
    classes_to_capture = os.listdir(directory)
    # classes_to_capture = ['Normal','Yawning']
    idx = np.arange(len(classes_to_capture))
    class_idx = dict(zip(classes_to_capture, idx))

    all_windows = []
    all_face_windows = []
    all_labels = []

    r_ear_minmax = [float('inf'), float('-inf')]
    l_ear_minmax = [float('inf'), float('-inf')]
    mar_minmax = [float('inf'), float('-inf')]
    phi_minmax = [float('inf'), float('-inf')]
    theta_minmax = [float('inf'), float('-inf')]

    for fol in tqdm(classes_to_capture):
        # print(fol)
        for videos in tqdm(os.listdir(os.path.join(directory, fol))):
            cap = cv2.VideoCapture(os.path.join(directory, fol, videos), cv2.CAP_FFMPEG)
            if (cap.isOpened() == False):
                print("Error opening video stream or file")
            # Read until video is completed
            # cnt=0
            face_imgs = []
            r_ear_li = []
            l_ear_li = []
            mar_li = []
            phi_li = []
            theta_li = []
            f = 0
            while (cap.isOpened()):
                # Capture frame-by-frame
                # print(cnt)
                ret, frame = cap.read()
                # print(directory,fol,videos)
                # f = frame
                if ret == True:
                    # STEP 3: Load the input image.
                    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                    # STEP 4: Detect face landmarks from the input image.
                    detection_result = detector.detect(image)
                    # STEP 5: Process the detection result. In this case, visualize it.
                    # annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
                    # cv2.imshow('img',cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
                    try:
                        landmarks = np.array([(face_landmarks.x, face_landmarks.y) for face_landmarks in
                                              detection_result.face_landmarks[0]])
                    except:
                        print('No landmark', fol, videos)
                        # cv2.imshow('frame',frame)
                        # cv2.waitKey(0)
                        continue
                    r_ear = calculate_ear(landmarks, [160, 144, 159, 145, 158, 153, 33, 133])
                    # Left Eye Aspect Ratio (L-EAR) calculation
                    l_ear = calculate_ear(landmarks, [385, 380, 386, 374, 387, 373, 362, 263])
                    # Mouth Aspect Ratio (MAR) calculation
                    mar = calculate_mar(landmarks, [81, 178, 13, 14, 311, 402, 78, 308])
                    # Calculate head pose
                    phi, theta = calculate_head_pose(frame,
                                                     np.array([landmarks[i] for i in [10, 33, 263, 152, 61, 291]]))
                    # 👇 新增：保存当前帧图像（裁剪出人脸区域）
                    face_img = extract_face_from_frame(frame, landmarks)  # 需要定义 extract_face_from_frame 函数
                    face_imgs.append(face_img)

                    r_ear_li.append(r_ear)
                    l_ear_li.append(l_ear)
                    mar_li.append(mar)
                    phi_li.append(phi[0])
                    theta_li.append(theta[0])
                else:
                    # print("读取视频失败：" + videos)
                    # continue
                    break
                # cnt+=1
            cap.release()
            if len(r_ear_li) == 0:
                print("r_ear_li是空：" + videos)
                continue
            r_ear_minmax[0] = min(min(r_ear_li), r_ear_minmax[0])
            l_ear_minmax[0] = min(min(l_ear_li), l_ear_minmax[0])
            mar_minmax[0] = min(min(mar_li), mar_minmax[0])
            phi_minmax[0] = min(min(phi_li), phi_minmax[0])
            theta_minmax[0] = min(min(theta_li), theta_minmax[0])

            r_ear_minmax[1] = max(max(r_ear_li), r_ear_minmax[1])
            l_ear_minmax[1] = max(max(l_ear_li), l_ear_minmax[1])
            mar_minmax[1] = max(max(mar_li), mar_minmax[1])
            phi_minmax[1] = max(max(phi_li), phi_minmax[1])
            theta_minmax[1] = max(max(theta_li), theta_minmax[1])

            all_features = np.vstack([r_ear_li, l_ear_li, mar_li, phi_li, theta_li])
            windows, face_windows, labels = create_windows(all_features,
                                                           all_face_imgs=face_imgs,
                                                           label=fol,
                                                           class_idx=class_idx,
                                                           window_size=config['window_size'],
                                                           overlap=config['overlap'])

            all_windows.extend(windows)
            all_face_windows.extend(face_windows)
            all_labels.extend(labels)

    windows_arr = np.array(all_windows)
    labels_array = np.array(all_labels)
    all_face_windows = np.array(all_face_windows)

    windows_arr[:, 0, :] = (windows_arr[:, 0, :] - r_ear_minmax[0]) / (r_ear_minmax[1] - r_ear_minmax[0])
    windows_arr[:, 1, :] = (windows_arr[:, 1, :] - l_ear_minmax[0]) / (l_ear_minmax[1] - l_ear_minmax[0])
    windows_arr[:, 2, :] = (windows_arr[:, 2, :] - mar_minmax[0]) / (mar_minmax[1] - mar_minmax[0])
    windows_arr[:, 3, :] = (windows_arr[:, 3, :] - phi_minmax[0]) / (phi_minmax[1] - phi_minmax[0])
    windows_arr[:, 4, :] = (windows_arr[:, 4, :] - theta_minmax[0]) / (theta_minmax[1] - theta_minmax[0])

    norm = np.vstack([r_ear_minmax, l_ear_minmax, mar_minmax, phi_minmax, theta_minmax])
    np.save(config['normalize_data_file'], norm)
    print("------------Norm params shape:", norm.shape)
    print("------------Norm params:\n", norm)
    np.save(config['preprocessed_windows_data'], windows_arr)
    np.save(config['preprocessed_labels_data'], labels_array)
    np.save(config['preprocessed_face_windows_data'], all_face_windows)
    return class_idx, windows_arr, all_face_windows, labels_array


def extract_face_from_frame(frame, landmarks, padding=0.2):
    """
    根据关键点裁剪人脸区域
    :param frame: 当前帧图像 (np.ndarray)
    :param landmarks: 归一化坐标关键点 (np.ndarray)
    :param padding: 扩展边界比例
    :return: 裁剪后的人脸图像
    """
    h, w, _ = frame.shape
    landmarks_abs = np.copy(landmarks)
    landmarks_abs[:, 0] *= w  # x -> pixel
    landmarks_abs[:, 1] *= h  # y -> pixel
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
