from torch.utils.data import Dataset
import torch


class CustomDataset(Dataset):
    def __init__(self, geo_data, face_data, labels):
        """
               :param geo_data: NumPy array, 几何特征 (N, ...)
               :param face_data: NumPy array, 面部图像 (N, T, H, W, C)
               :param labels: NumPy array, 标签 (N,)
               """
        self.geo_data = geo_data  # 保持为 NumPy 数组
        self.face_data = face_data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # 按需读取单个样本并转换为 Tensor
        geo = torch.tensor(self.geo_data[idx]).float()
        face = torch.tensor(self.face_data[idx]).float()  # shape: (T, H, W, C)
        label = torch.tensor(self.labels[idx]).long()

        # 转换为 (T, C, H, W)
        face = face.permute(0, 3, 1, 2)  # → (T, C, H, W)

        return geo, face, label
