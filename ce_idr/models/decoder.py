import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderBlock(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):

    def __init__(self, chs):
        super().__init__()
        (c1, c2, c3, c4) = chs
        self.up3 = DecoderBlock(c4 + c3, c3)
        self.up2 = DecoderBlock(c3 + c2, c2)
        self.up1 = DecoderBlock(c2 + c1, c1)
        self.out = nn.Conv2d(c1, 1, 1)

    def forward(self, f4, f3, f2, f1):
        x = F.interpolate(f4, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.up3(torch.cat([x, f3], dim=1))
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.up2(torch.cat([x, f2], dim=1))
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.up1(torch.cat([x, f1], dim=1))
        return torch.sigmoid(self.out(x))
