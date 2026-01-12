import torch
from torch.utils.data import Dataset, DataLoader


class CustomDataset_old(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs)  # Number of samples

    def __getitem__(self, idx):
        # Return the input and corresponding label
        input_sample = self.inputs[idx]
        label_sample = self.labels[idx]
        return input_sample, label_sample
