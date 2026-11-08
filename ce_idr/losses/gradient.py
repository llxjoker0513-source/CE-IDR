import torch
import torch.nn.functional as F

def gradient(img):
    dx = img[:, :, :, 1:] - img[:, :, :, :-1]
    dy = img[:, :, 1:, :] - img[:, :, :-1, :]
    return (dx, dy)

def gradient_loss(fused, vis, ir):
    (fx, fy) = gradient(fused)
    (vx, vy) = gradient(vis)
    (ix, iy) = gradient(ir)
    target_x = torch.max(vx.abs(), ix.abs())
    target_y = torch.max(vy.abs(), iy.abs())
    return F.l1_loss(fx.abs(), target_x) + F.l1_loss(fy.abs(), target_y)

def sobel_edges(img):
    channels = img.size(1)
    kx = img.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3)
    ky = img.new_tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3)
    kx = kx.repeat(channels, 1, 1, 1)
    ky = ky.repeat(channels, 1, 1, 1)
    gx = F.conv2d(img, kx, padding=1, groups=channels)
    gy = F.conv2d(img, ky, padding=1, groups=channels)
    return torch.sqrt(gx * gx + gy * gy + 1e-06)

def sobel_edge_loss(fused, vis, ir):
    fused_edge = sobel_edges(fused)
    target_edge = torch.max(sobel_edges(vis), sobel_edges(ir))
    return F.l1_loss(fused_edge, target_edge)
