import numpy as np

# 替换为你目录中的 .npy 文件路径
file_path = r'G:\A_wangft_bs\5G的npy文件\preprocess_label.npy'

# 加载 .npy 文件
data = np.load(file_path)

# 输出数组的形状、数据类型和总数据量
print(f'数组形状: {data.shape}')
print(f'数据类型: {data.dtype}')
print(f'总数据量: {data.size}')
