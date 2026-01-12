import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import cv2
import numpy as np
from tqdm import tqdm

from config_parser import parse_file
from tools.ear_mar_pose import calculate_ear, calculate_mar, calculate_head_pose
from tools.extract_face_frame import extract_face_from_frame


def get_detector(model_path):
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                           output_face_blendshapes=True,
                                           output_facial_transformation_matrixes=True,
                                           num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)
    return detector


def dynamic_step_create_windows(video_file, all_features, all_face_imgs, all_labels,
                                window_size=10, normal_step=6, keyframe_step=4):
    """
    动态滑动窗口策略：
    - 第一个窗口固定为 [0, 10)
    - 初始步长为 6，检测是否有闭眼/打哈欠帧
    - 若有，则进入“关键帧模式”使用步长 2，根据规则构建窗口
    - 若连续 4 帧正常，恢复大步长 6
    - 最后一帧不够 10 帧则向前取 10 帧作为最后窗口
    """

    windows = []
    face_windows = []
    labels = []

    num_frames = len(all_labels)
    i = 0
    current_step = normal_step
    normal_counter = 0  # 连续正常帧计数器

    # 🔹 第一个窗口固定从 0 开始
    if num_frames >= window_size:
        windows.append(all_features[:, 0:window_size])
        face_windows.append(all_face_imgs[0:window_size])
        labels.append(all_labels[window_size - 1])
        i = 0 + normal_step
    else:
        print("⚠️ 帧数不足，无法构建第一个窗口")
        return [], [], []

    while i <= num_frames:
        # 当前窗口结束位置
        end = i + window_size
        if end > num_frames:
            # 如果当前窗口越界了，但还能从最后 window_size 帧取一个窗口
            if num_frames >= window_size:
                last_window = all_features[:, -window_size:]
                last_face_window = all_face_imgs[-window_size:]
                windows.append(last_window)
                face_windows.append(last_face_window)
                labels.append(all_labels[-1])  # 标签取最后一帧
            break

        window_labels = all_labels[i:end]
        has_keyframe = any(label in [0, 2] for label in window_labels)

        if has_keyframe:
            # 找到第一个关键帧 j
            first_keyframe_idx = None
            for j in range(i, end):
                if all_labels[j] in [0, 2]:
                    first_keyframe_idx = j
                    break

            if first_keyframe_idx is not None:
                win_start = first_keyframe_idx - window_size + 1
                # 如果前面帧不够 → 需要前向填充
                if win_start < 0:
                    missing = window_size - (first_keyframe_idx + 1)
                    first_valid_frame = all_features[:, :1]  # 取第一个有效帧
                    padded_window = np.hstack([
                        np.repeat(first_valid_frame, missing, axis=1),
                        all_features[:, :first_keyframe_idx + 1]
                    ])
                    padded_face_window = [all_face_imgs[0]] * missing + all_face_imgs[:first_keyframe_idx + 1]
                    windows.append(padded_window)
                    face_windows.append(padded_face_window)
                    labels.append(all_labels[first_keyframe_idx])
                else:
                    window = all_features[:, win_start:first_keyframe_idx + 1]
                    face_window = all_face_imgs[win_start:first_keyframe_idx + 1]
                    windows.append(window)
                    face_windows.append(face_window)
                    labels.append(all_labels[first_keyframe_idx])
            # i = i + keyframe_step
            current_step = keyframe_step
            normal_counter = 0
        else:
            # 检查当前两帧是否都正常
            next_two_labels = all_labels[i:i + 2] if i + 2 <= num_frames else all_labels[i:]
            if len(next_two_labels) == 2 and next_two_labels[0] == 1 and next_two_labels[1] == 1:
                normal_counter += 2
                if normal_counter >= 4:
                    # 连续 6 帧正常 → 回到 normal_step = 6
                    window = all_features[:, i:i + window_size]
                    face_window = all_face_imgs[i:i + window_size]
                    windows.append(window)
                    face_windows.append(face_window)
                    labels.append(all_labels[i + window_size - 1])
                    normal_counter = 0
                    current_step = normal_step
                # i += keyframe_step
            else:
                # 其中至少有一帧是关键帧 → 构建窗口
                win_end = min(i + window_size, num_frames)
                window = all_features[:, i:win_end]
                face_window = all_face_imgs[i:win_end]
                windows.append(window)
                face_windows.append(face_window)
                labels.append(all_labels[win_end - 1])
                current_step = keyframe_step
                normal_counter = 0
                # i += keyframe_step
        i += current_step

    # 统计闭眼、正常、打哈欠的窗口数量
    from collections import Counter
    label_counter = Counter(labels)
    print(f" 🎬 Video: {video_file} |  📊 Label distribution:")
    print(f" 🎬 Video: {video_file} |  👁️ 闭眼（0）: {label_counter.get(0, 0)} 个窗口")
    print(f" 🎬 Video: {video_file} |  ✅ 正常（1）: {label_counter.get(1, 0)} 个窗口")
    print(f" 🎬 Video: {video_file} |  😴 打哈欠（2）: {label_counter.get(2, 0)} 个窗口")
    return windows, face_windows, labels


def preprocess_data(config):
    detector = get_detector(config['mediapipe_model_path'])
    directory = config['train_dataset_path']
    video_root = config['train_dataset_path']

    all_windows = []
    all_face_windows = []
    all_labels = []

    # 存储每个特征的 sum 和 sum_squares 用于计算 mean 和 std
    feature_stats = {
        'r_ear': {'sum': 0.0, 'sum_sq': 0.0, 'count': 0},
        'l_ear': {'sum': 0.0, 'sum_sq': 0.0, 'count': 0},
        'mar': {'sum': 0.0, 'sum_sq': 0.0, 'count': 0},
        'phi': {'sum': 0.0, 'sum_sq': 0.0, 'count': 0},
        'theta': {'sum': 0.0, 'sum_sq': 0.0, 'count': 0}
    }
    # 获取所有视频文件（不考虑子目录）
    video_files = [f for f in os.listdir(video_root) if f.endswith(('.mp4', '.avi', '.mov'))]
    # 🟢 添加全局进度条
    pbar = tqdm(total=len(video_files), desc="📊 Overall Progress", position=0)
    for video_file in video_files:
        video_path = os.path.join(video_root, video_file)
        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            print("Error opening video stream or file")
            pbar.update(1)  # 🟢 更新进度条
        # 🔍 获取总帧数
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"🎬 Video: {video_file} | Total frames: {total_frames}")
        # Read until video is completed
        # cnt=0
        face_imgs = []
        r_ear_li = []
        l_ear_li = []
        mar_li = []
        phi_li = []
        theta_li = []
        labels_li = []
        f = 0
        while cap.isOpened():
            # Capture frame-by-frame
            # print(cnt)
            ret, frame = cap.read()
            # print(directory,fol,videos)
            # f = frame
            if ret:
                f += 1  # 每读取一帧就加 1
                h, w, _ = frame.shape
                # STEP 3: Load the input image.
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                # STEP 4: Detect face landmarks from the input image.
                detection_result = detector.detect(image)
                # STEP 5: Process the detection result. In this case, visualize it.
                # annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
                # cv2.imshow('img',cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
                try:
                    landmarks = np.array([(face_landmarks.x, face_landmarks.y) for face_landmarks in
                                          detection_result.face_landmarks[0]])
                except:
                    print(f'---------No landmark---------- Frame #{f} has no landmarks')  # cv2.imshow('frame',frame)
                    # cv2.waitKey(0)
                    continue
                r_ear = calculate_ear(landmarks, [160, 144, 159, 145, 158, 153, 33, 133])
                # Left Eye Aspect Ratio (L-EAR) calculation
                l_ear = calculate_ear(landmarks, [385, 380, 386, 374, 387, 373, 362, 263])
                ear = (r_ear + l_ear) / 2.0
                # Mouth Aspect Ratio (MAR) calculation
                mar = calculate_mar(landmarks, [81, 178, 13, 14, 311, 402, 78, 308])
                # Head pose
                landmarks_2d = np.array([landmarks[i] for i in [10, 33, 263, 152, 61, 291]])
                landmarks_2d[:, 0] *= w
                landmarks_2d[:, 1] *= h
                phi, theta = calculate_head_pose(frame, landmarks_2d)

                # 👇 新增：保存当前帧图像（裁剪出人脸区域）
                face_img = extract_face_from_frame(frame, landmarks)  # 需要定义 extract_face_from_frame 函数
                #cv2.imwrite("face.png", (face_img * 255).astype(np.uint8))  # 恢复到 [0,255]
                #face_imgs.append(face_img)
                face_imgs.append(face_img.astype(np.float32))  # 从 float64 改成 float32

                r_ear_li.append(r_ear)
                l_ear_li.append(l_ear)
                mar_li.append(mar)
                phi_li.append(phi)
                theta_li.append(theta)

                # 判断标签（MAR > 0.7 → 2；EAR < 0.2 → 0）
                if mar > 0.7:
                    label = 2
                # elif right_ear < 0.2 or left_ear < 0.2:
                elif r_ear < 0.2 or l_ear < 0.2:
                    label = 0
                else:
                    label = 1
                labels_li.append(label)
            else:
                # print("读取视频失败：" + videos)
                # continue
                break
            # cnt+=1
        cap.release()
        if len(r_ear_li) == 0:
            print("r_ear_li是空：" + video_file)
            continue
        assert len(r_ear_li) == len(face_imgs) == len(labels_li), \
            f"Length mismatch: features={len(r_ear_li)}, face_imgs={len(face_imgs)}, labels={len(labels_li)}"
        print(f"✅ r_ear_li length: {len(r_ear_li)}")
        # 累加统计信息
        for r, l, m, p, t in zip(r_ear_li, l_ear_li, mar_li, phi_li, theta_li):
            feature_stats['r_ear']['sum'] += r
            feature_stats['r_ear']['sum_sq'] += r ** 2
            feature_stats['r_ear']['count'] += 1

            feature_stats['l_ear']['sum'] += l
            feature_stats['l_ear']['sum_sq'] += l ** 2
            feature_stats['l_ear']['count'] += 1

            feature_stats['mar']['sum'] += m
            feature_stats['mar']['sum_sq'] += m ** 2
            feature_stats['mar']['count'] += 1

            feature_stats['phi']['sum'] += p
            feature_stats['phi']['sum_sq'] += p ** 2
            feature_stats['phi']['count'] += 1

            feature_stats['theta']['sum'] += t
            feature_stats['theta']['sum_sq'] += t ** 2
            feature_stats['theta']['count'] += 1

        all_features = np.vstack([r_ear_li, l_ear_li, mar_li, phi_li, theta_li])

        windows, face_windows, labels = dynamic_step_create_windows(video_file, all_features=all_features,
                                                                    all_face_imgs=face_imgs,
                                                                    all_labels=labels_li,
                                                                    window_size=config[
                                                                        'window_size'])  # normal_step = config['overlap']
        # ✅ 打印当前视频生成的窗口数量
        print(
            f"🧮 Video: {video_file} | Generated windows: {len(windows)}| Generated face_windows: {len(face_windows)}| Generated labels: {len(labels)}")

        all_windows.extend(windows)
        all_face_windows.extend(face_windows)
        all_labels.extend(labels)
        pbar.update(1)  # 🟢 更新进度条：每处理完一个视频就 +1
    pbar.close()  # 🟢 关闭进度条
    # 计算均值和标准差
    means = []
    stds = []
    for feat in ['r_ear', 'l_ear', 'mar', 'phi', 'theta']:
        count = feature_stats[feat]['count']
        mean = feature_stats[feat]['sum'] / count
        std = np.sqrt(feature_stats[feat]['sum_sq'] / count - mean ** 2)
        means.append(mean)
        stds.append(std)

    from collections import Counter
    all_label_counter = Counter(all_labels)
    print(f"📊 总的 Label distribution:")
    print(f"   👁️ 总的闭眼（0）: {all_label_counter.get(0, 0)} 个窗口")
    print(f"   ✅ 总的正常（1）: {all_label_counter.get(1, 0)} 个窗口")
    print(f"   😴 总的打哈欠（2）: {all_label_counter.get(2, 0)} 个窗口")
    windows_arr = np.array(all_windows)
    labels_array = np.array(all_labels)
    all_face_windows = np.array(all_face_windows)

    # Z-Score 归一化
    for i in range(windows_arr.shape[1]):
        windows_arr[:, i, :] = (windows_arr[:, i, :] - means[i]) / (stds[i] + 1e-8)  # 防止除以零
    # 保存归一化参数
    norm_params = np.array([means, stds])
    np.save(config['normalize_data_file'], norm_params)
    print("------------Norm params shape:", norm_params.shape)
    print("------------Norm params:\n", norm_params)
    np.save(config['preprocessed_windows_data'], windows_arr)
    np.save(config['preprocessed_labels_data'], labels_array)
    np.save(config['preprocessed_face_windows_data'], all_face_windows)
    return windows_arr, all_face_windows, labels_array


config_file = parse_file('config.ini')
windows_arr1, all_face_windows1, labels_array1 = preprocess_data(config_file)
