import torch
import torch.nn as nn
import torch.nn.functional as F

class CSRM(nn.Module):

    def __init__(self, low_ch, sem_ch):
        super().__init__()
        self.gamma = nn.Conv2d(sem_ch, low_ch, 1)
        self.beta = nn.Conv2d(sem_ch, low_ch, 1)

    def forward(self, low_feat, high_sem_feat):
        high_sem_feat = F.interpolate(high_sem_feat, size=low_feat.shape[-2:], mode='bilinear', align_corners=False)
        g = self.gamma(high_sem_feat)
        b = self.beta(high_sem_feat)
        return g * low_feat + b
