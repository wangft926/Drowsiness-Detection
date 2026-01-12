import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: shape (N, C) 或 (N, C, d1, d2, ..., dk) -> logits
        targets: shape (N,) 或 (N, d1, d2, ..., dk) -> ground truth class indices
        """
        log_prob = -F.cross_entropy(inputs, targets, reduction='none')
        prob = torch.exp(log_prob)

        focal_weight = self.alpha * (1 - prob) ** self.gamma
        loss = -focal_weight * log_prob

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
