import os
import pandas as pd

# 读取CSV文件
csv_file = r'G:\A_wangft_bs\Drowsiness-Detection-using-CNN-and-LSTM-main\test\fatigue_detection_results.csv'
df = pd.read_csv(csv_file)

# 获取视频名称列
video_names = df['video_name'].tolist()

# 指定要遍历的目录
directory = r'G:\A-数据集\YawDD\test'

# 遍历目录中的文件
for filename in os.listdir(directory):
    file_path = os.path.join(directory, filename)

    # 如果文件不在CSV的第一列中，则删除
    if filename not in video_names:
        os.remove(file_path)
        print(f"Deleted: {file_path}")
