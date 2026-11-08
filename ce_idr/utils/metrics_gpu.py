import torch
import torch.nn.functional as F
import math

def _check(x):
    assert x.dim() == 4 and x.size(1) == 1, 'Input must be [B,1,H,W]'
    return x

def _normalize(x):
    minv = x.amin(dim=[2, 3], keepdim=True)
    maxv = x.amax(dim=[2, 3], keepdim=True)
    return (x - minv) / (maxv - minv + 1e-08)

def entropy_batch(x, num_bins=256):
    _check(x)
    B = x.size(0)
    xq = torch.clamp((x * (num_bins - 1)).long(), 0, num_bins - 1)
    ent = []
    for b in range(B):
        hist = torch.bincount(xq[b].view(-1), minlength=num_bins).float()
        p = hist / hist.sum()
        p = p[p > 0]
        ent.append(-(p * torch.log2(p)).sum())
    return torch.stack(ent)

def standard_deviation_batch(x):
    _check(x)
    x = x.view(x.size(0), -1)
    return x.std(dim=1, unbiased=False)

def mutual_information_batch(x, y, num_bins=256):
    _check(x)
    _check(y)
    B = x.size(0)
    xq = torch.clamp((x * (num_bins - 1)).long(), 0, num_bins - 1)
    yq = torch.clamp((y * (num_bins - 1)).long(), 0, num_bins - 1)
    mi = []
    for b in range(B):
        joint = torch.zeros(num_bins, num_bins, device=x.device)
        joint.index_put_((xq[b].view(-1), yq[b].view(-1)), torch.ones_like(xq[b].view(-1), dtype=torch.float), accumulate=True)
        pxy = joint / joint.sum()
        px = pxy.sum(dim=1, keepdim=True)
        py = pxy.sum(dim=0, keepdim=True)
        nz = pxy > 0
        mi.append((pxy[nz] * torch.log2(pxy[nz] / (px @ py)[nz])).sum())
    return torch.stack(mi)

def fusion_mi_batch(fused, vis, ir):
    return mutual_information_batch(fused, vis) + mutual_information_batch(fused, ir)

def _gaussian_kernel(kernel_size=11, sigma=1.5, channels=1, device='cuda'):
    coords = torch.arange(kernel_size, device=device).float() - kernel_size // 2
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    kernel1d = g.view(1, 1, -1)
    kernel2d = kernel1d.transpose(1, 2) @ kernel1d
    kernel2d = kernel2d.expand(channels, 1, kernel_size, kernel_size)
    return kernel2d

def ssim_batch(x, y, kernel_size=11, sigma=1.5):
    _check(x)
    _check(y)
    device = x.device
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    kernel = _gaussian_kernel(kernel_size, sigma, 1, device)
    mu_x = F.conv2d(x, kernel, padding=kernel_size // 2)
    mu_y = F.conv2d(y, kernel, padding=kernel_size // 2)
    mu_x2 = mu_x ** 2
    mu_y2 = mu_y ** 2
    mu_xy = mu_x * mu_y
    sigma_x2 = F.conv2d(x * x, kernel, padding=kernel_size // 2) - mu_x2
    sigma_y2 = F.conv2d(y * y, kernel, padding=kernel_size // 2) - mu_y2
    sigma_xy = F.conv2d(x * y, kernel, padding=kernel_size // 2) - mu_xy
    ssim_map = (2 * mu_xy + C1) * (2 * sigma_xy + C2) / ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2) + 1e-08)
    return ssim_map.mean(dim=[1, 2, 3])

def ms_ssim_batch(x, y, levels=5):
    weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    out = torch.ones(x.size(0), device=x.device)
    for i in range(levels):
        ssim_map = ssim_batch(x, y)
        out *= ssim_map ** weights[i]
        x = F.avg_pool2d(x, 2)
        y = F.avg_pool2d(y, 2)
    return out

def fusion_ssim_batch(fused, vis, ir):
    return 0.5 * (ssim_batch(fused, vis) + ssim_batch(fused, ir))

def fusion_ms_ssim_batch(fused, vis, ir):
    return 0.5 * (ms_ssim_batch(fused, vis) + ms_ssim_batch(fused, ir))

def cc_batch(x, y):
    _check(x)
    _check(y)
    x = x.view(x.size(0), -1)
    y = y.view(y.size(0), -1)
    vx = x - x.mean(dim=1, keepdim=True)
    vy = y - y.mean(dim=1, keepdim=True)
    return (vx * vy).sum(dim=1) / (torch.sqrt((vx ** 2).sum(dim=1)) * torch.sqrt((vy ** 2).sum(dim=1)) + 1e-08)

def fusion_cc_batch(fused, vis, ir):
    return 0.5 * (cc_batch(fused, vis) + cc_batch(fused, ir))

def scd_batch(fused, vis, ir):
    return fusion_cc_batch(fused, vis, ir) * 2 - cc_batch(vis, ir)

def spatial_frequency_batch(x):
    _check(x)
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    RF = torch.sqrt((dx ** 2).mean(dim=[1, 2, 3]))
    CF = torch.sqrt((dy ** 2).mean(dim=[1, 2, 3]))
    return torch.sqrt(RF ** 2 + CF ** 2)

def average_gradient_batch(x):
    _check(x)
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    g = torch.sqrt(dx[:, :, :-1, :] ** 2 + dy[:, :, :, :-1] ** 2)
    return g.mean(dim=[1, 2, 3])

def _sobel_kernel(device):
    kx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], device=device, dtype=torch.float32).view(1, 1, 3, 3)
    ky = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], device=device, dtype=torch.float32).view(1, 1, 3, 3)
    return (kx, ky)

def qabf_batch(fused, vis, ir):
    _check(fused)
    _check(vis)
    _check(ir)
    device = fused.device
    (kx, ky) = _sobel_kernel(device)

    def grad(x):
        gx = F.conv2d(x, kx, padding=1)
        gy = F.conv2d(x, ky, padding=1)
        g = torch.sqrt(gx ** 2 + gy ** 2 + 1e-08)
        a = torch.atan2(gy, gx + 1e-08)
        return (g, a)
    (gF, aF) = grad(fused)
    (gV, aV) = grad(vis)
    (gI, aI) = grad(ir)
    Qg_V = (2 * gV * gF + 1e-08) / (gV ** 2 + gF ** 2 + 1e-08)
    Qg_I = (2 * gI * gF + 1e-08) / (gI ** 2 + gF ** 2 + 1e-08)
    Qa_V = torch.cos(aV - aF)
    Qa_I = torch.cos(aI - aF)
    Q_V = Qg_V * Qa_V
    Q_I = Qg_I * Qa_I
    wV = gV / (gV + gI + 1e-08)
    wI = gI / (gV + gI + 1e-08)
    Q = wV * Q_V + wI * Q_I
    return Q.mean(dim=[1, 2, 3])

def _gaussian_kernel_2d(kernel_size, sigma, device):
    coords = torch.arange(kernel_size, device=device).float() - kernel_size // 2
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    kernel1d = g.view(1, 1, -1)
    kernel2d = kernel1d.transpose(1, 2) @ kernel1d
    return kernel2d

def _gaussian_blur(x, kernel_size, sigma):
    device = x.device
    kernel = _gaussian_kernel_2d(kernel_size, sigma, device)
    kernel = kernel.expand(x.size(1), 1, kernel_size, kernel_size)
    return F.conv2d(x, kernel, padding=kernel_size // 2, groups=x.size(1))

def vif_batch(ref, dist):
    _check(ref)
    _check(dist)
    sigma_nsq = 2.0
    eps = 1e-10
    num = 0.0
    den = 0.0
    for scale in range(4):
        kernel_size = 2 ** (4 - scale) + 1
        sigma = kernel_size / 5.0
        ref_blur = _gaussian_blur(ref, kernel_size, sigma)
        dist_blur = _gaussian_blur(dist, kernel_size, sigma)
        mu1 = ref_blur
        mu2 = dist_blur
        sigma1_sq = _gaussian_blur(ref * ref, kernel_size, sigma) - mu1 ** 2
        sigma2_sq = _gaussian_blur(dist * dist, kernel_size, sigma) - mu2 ** 2
        sigma12 = _gaussian_blur(ref * dist, kernel_size, sigma) - mu1 * mu2
        sigma1_sq = torch.clamp(sigma1_sq, min=0)
        sigma2_sq = torch.clamp(sigma2_sq, min=0)
        g = sigma12 / (sigma1_sq + eps)
        sv_sq = sigma2_sq - g * sigma12
        g[sigma1_sq < eps] = 0
        sv_sq[sigma1_sq < eps] = sigma2_sq[sigma1_sq < eps]
        sigma1_sq[sigma1_sq < eps] = 0
        sv_sq = torch.clamp(sv_sq, min=eps)
        num += torch.sum(torch.log10(1 + g ** 2 * sigma1_sq / (sv_sq + sigma_nsq)), dim=[1, 2, 3])
        den += torch.sum(torch.log10(1 + sigma1_sq / sigma_nsq), dim=[1, 2, 3])
        ref = F.avg_pool2d(ref, 2)
        dist = F.avg_pool2d(dist, 2)
    return num / den

def fusion_vif_batch(fused, vis, ir):
    return 0.5 * (vif_batch(vis, fused) + vif_batch(ir, fused))

def evaluate_all_batch(fused, vis, ir):
    return {'EN': entropy_batch(fused), 'MI': fusion_mi_batch(fused, vis, ir), 'AG': average_gradient_batch(fused), 'SD': standard_deviation_batch(fused), 'SCD': scd_batch(fused, vis, ir), 'Qabf': qabf_batch(fused, vis, ir), 'VIF': fusion_vif_batch(fused, vis, ir), 'SSIM': fusion_ssim_batch(fused, vis, ir), 'MS_SSIM': fusion_ms_ssim_batch(fused, vis, ir), 'CC': fusion_cc_batch(fused, vis, ir), 'SF': spatial_frequency_batch(fused)}

def evaluate_main_batch(fused, vis, ir):
    return {'EN': entropy_batch(fused), 'AG': average_gradient_batch(fused), 'SD': standard_deviation_batch(fused), 'SCD': scd_batch(fused, vis, ir), 'Qabf': qabf_batch(fused, vis, ir), 'VIF': fusion_vif_batch(fused, vis, ir), 'SF': spatial_frequency_batch(fused)}
