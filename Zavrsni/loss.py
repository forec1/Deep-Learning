import torch.nn as nn
import torch


class ReverseHuberLoss(nn.Module):

    def forward(self, input, target):
        diff = torch.abs(input - target)
        c = 0.2 * torch.max(diff)
        loss = torch.where(diff <= c, diff, (diff**2 + c**2) / (2 * c))
        loss = loss.sum()
        return loss / len(diff)
