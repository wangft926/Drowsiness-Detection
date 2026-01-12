import os
import cv2

# 指定目录路径
directory = r'G:\A_wangft_bs\5G的npy的dataset'

# 初始化总帧数
total_frames = 0

# 遍历目录下的所有视频文件
for filename in os.listdir(directory):
    if filename.endswith(('.mp4', '.avi', '.mkv')):
        file_path = os.path.join(directory, filename)
        cap = cv2.VideoCapture(file_path)
        if cap.isOpened():
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            total_frames += frame_count
            print(f'文件: {filename}, 帧数: {frame_count}')
            cap.release()
        else:
            print(f'无法打开文件: {filename}')

# 输出总帧数
print(f'所有视频总帧数: {total_frames}')
