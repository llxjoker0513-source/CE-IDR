import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):

    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(in_ch, out_ch, k, s, p), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.net(x)

class StructuralEncoder(nn.Module):

    def __init__(self, in_ch=None, base_ch=None):
        super().__init__()
        self.stage1 = ConvBlock(in_ch, base_ch)
        self.stage2 = ConvBlock(base_ch, base_ch * 2, s=2)
        self.stage3 = ConvBlock(base_ch * 2, base_ch * 4, s=2)
        self.stage4 = ConvBlock(base_ch * 4, base_ch * 8, s=2)

    def forward(self, x):
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
        return [f1, f2, f3, f4]
import torch
import torch.nn as nn
import torch.nn.functional as F

class SemanticBlock(nn.Module):

    def __init__(self, dim, expand=None):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.pwconv1 = nn.Conv2d(dim, dim * expand, 1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(dim * expand, dim, 1)
        self.norm = nn.BatchNorm2d(dim)

    def forward(self, x):
        y = self.dwconv(x)
        y = self.act(self.pwconv1(y))
        y = self.pwconv2(y)
        return self.norm(x + y)

class SemanticEncoder(nn.Module):

    def __init__(self, in_ch=None, base=None, expand=None):
        super().__init__()
        self.proj1 = nn.Conv2d(in_ch, base, 3, 1, 1)
        self.block1 = SemanticBlock(base, expand)
        self.down2 = nn.Conv2d(base, base * 2, 3, 2, 1)
        self.block2 = SemanticBlock(base * 2, expand)
        self.down3 = nn.Conv2d(base * 2, base * 4, 3, 2, 1)
        self.block3 = SemanticBlock(base * 4, expand)
        self.down4 = nn.Conv2d(base * 4, base * 8, 3, 2, 1)
        self.block4 = SemanticBlock(base * 8, expand)

    def forward(self, x):
        f1 = self.block1(F.relu(self.proj1(x)))
        f2 = self.block2(F.relu(self.down2(f1)))
        f3 = self.block3(F.relu(self.down3(f2)))
        f4 = self.block4(F.relu(self.down4(f3)))
        return [f1, f2, f3, f4]
