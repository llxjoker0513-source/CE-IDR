import torch
import torch.nn as nn
import torch.nn.functional as F

class UncertaintyHead(nn.Module):

    def __init__(self, ch):
        super().__init__()
        self.vis_head = nn.Conv2d(ch, 1, 1)
        self.ir_head = nn.Conv2d(ch, 1, 1)

    def forward(self, x):
        sigma_vis = F.softplus(self.vis_head(x))
        sigma_ir = F.softplus(self.ir_head(x))
        return (sigma_vis, sigma_ir)

def uncertainty_arbitration(fused_vis, fused_ir, sigma_vis, sigma_ir):
    c_vis = torch.exp(-sigma_vis)
    c_ir = torch.exp(-sigma_ir)
    out = (c_vis * fused_vis + c_ir * fused_ir) / (c_vis + c_ir + 1e-06)
    return out
