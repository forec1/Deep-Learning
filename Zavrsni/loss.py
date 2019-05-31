import torch.nn as nn
import torch


class ReverseHuberLoss(nn.Module):

    def forward(self, truth, target, device):
        label_mask = (truth > 0).float()
        n_val_pixels = torch.sum(label_mask, dtype=torch.float)
        diff = (truth - target)*label_mask
        c = 0.2 * torch.max(diff)
        loss = torch.where(diff <= c, torch.abs(diff), (diff**2 + c**2) / (2 * c))
        loss = torch.sum(loss)
        loss = loss / n_val_pixels
        return loss
