import os
import pandas as pd
from collections import defaultdict

# 读取CSV文件
csv_file = r'G:\A_wangft_bs\Drowsiness-Detection-using-CNN-and-LSTM-main\test\fatigue_detection_results.csv'
df = pd.read_csv(csv_file)

# 获取视频名称列
video_names = df['video_name'].tolist()

# 指定要遍历的目录
directory = r'G:\A-数据集\YawDD\test'

# 遍历目录中的文件
test_files = []
file_extensions = defaultdict(int)  # 用于统计各种后缀的文件数量

for root, dirs, files in os.walk(directory):
    for file in files:
        test_files.append(file)
        # 获取文件扩展名并统计
        ext = os.path.splitext(file)[1]
        file_extensions[ext] += 1

# 对比两个列表，找出test目录中缺少的视频
missing_videos = set(video_names) - set(test_files)
# 同时也检查test目录中多余的文件
extra_videos = set(test_files) - set(video_names)

print(f"CSV中视频总数: {len(video_names)}")
print(f"test目录中文件数: {len(test_files)}")
print(f"test目录中缺少的视频数量: {len(missing_videos)}")
print(f"test目录中额外的视频数量: {len(extra_videos)}")

# 打印各种后缀文件的统计信息
print("\ntest目录中各后缀文件数量:")
for ext, count in sorted(file_extensions.items()):
    print(f"{ext}: {count}")

# 分别统计三种特定后缀文件的数量
yawning_count = len([f for f in test_files if f.endswith('Yawning.avi')])
normal_count = len([f for f in test_files if f.endswith('Normal.avi')])
talking_count = len([f for f in test_files if f.endswith('Talking.avi')])

print(f"\n特定类型文件数量:")
print(f"Yawning.avi: {yawning_count}")
print(f"Normal.avi: {normal_count}")
print(f"Talking.avi: {talking_count}")

print("\n缺少的视频列表:")
for video in sorted(missing_videos):
    print(video)

print("\n额外的视频列表:")
for video in sorted(extra_videos):
    print(video)

# 检查特定的Yawning.avi文件
print(f"\nYawning.avi在CSV中: {'Yawning.avi' in video_names}")
print(f"Yawning.avi在test目录中: {'Yawning.avi' in test_files}")
