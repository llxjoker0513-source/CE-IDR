import torch
import torch.nn.functional as F
from .gradient import gradient_loss, sobel_edge_loss, sobel_edges

def ssim_loss(x, y):
    mu_x = x.mean()
    mu_y = y.mean()
    sigma_x = x.var()
    sigma_y = y.var()
    sigma_xy = ((x - mu_x) * (y - mu_y)).mean()
    (c1, c2) = (0.0001, 0.0001)
    ssim = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2))
    return 1 - ssim

class FusionLoss(torch.nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    @staticmethod
    def _normalize_map(x):
        min_v = x.amin(dim=(-2, -1), keepdim=True)
        max_v = x.amax(dim=(-2, -1), keepdim=True)
        return (x - min_v) / (max_v - min_v + 1e-06)

    @staticmethod
    def _quality_mean(x):
        return x.mean(dim=(1, 2, 3)).mean()

    def quality_scores(self, vis, ir):
        with torch.no_grad():
            vis_edge = sobel_edges(vis)
            ir_edge = sobel_edges(ir)
            target_edge = torch.max(vis_edge, ir_edge)
            vis_hf = torch.abs(vis - F.avg_pool2d(vis, 3, 1, 1))
            ir_local = ir - F.avg_pool2d(ir, 7, 1, 3)
            ir_saliency = F.relu(ir_local)
            detail_score = self._quality_mean(self._normalize_map(target_edge))
            texture_score = self._quality_mean(self._normalize_map(vis_hf))
            saliency_score = self._quality_mean(self._normalize_map(ir_saliency))
            gap = torch.abs(vis_edge - ir_edge).mean()
            edge_sum = (vis_edge + ir_edge).mean() + 1e-06
            agreement_score = torch.clamp(1.0 - gap / edge_sum, 0.0, 1.0)
        return {'struct': saliency_score, 'grad': texture_score, 'sobel': detail_score, 'ssim': agreement_score}

    def adaptive_lambda(self, name, score):
        base = self.cfg[name]
        if not self.cfg['use_quality_adaptive']:
            return base
        strength = self.cfg['qa_strength']
        min_scale = self.cfg['qa_min_scale']
        max_scale = self.cfg['qa_max_scale']
        scale = 1.0 + strength * (score - 0.5)
        scale = torch.clamp(scale, min_scale, max_scale)
        return base * scale

    @staticmethod
    def local_std(x, kernel_size=7):
        padding = kernel_size // 2
        mean = F.avg_pool2d(x, kernel_size, stride=1, padding=padding)
        mean_sq = F.avg_pool2d(x * x, kernel_size, stride=1, padding=padding)
        var = torch.clamp(mean_sq - mean * mean, min=0.0)
        return torch.sqrt(var + 1e-06)

    @staticmethod
    def _global_std(x):
        return torch.std(x, dim=(-2, -1), keepdim=True, unbiased=False)

    @staticmethod
    def _global_mad(x):
        mean = x.mean(dim=(-2, -1), keepdim=True)
        return torch.mean(torch.abs(x - mean), dim=(-2, -1), keepdim=True)

    @staticmethod
    def laplacian_response(x):
        channels = x.size(1)
        kernel = x.new_tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]).view(1, 1, 3, 3)
        kernel = kernel.repeat(channels, 1, 1, 1)
        return torch.abs(F.conv2d(x, kernel, padding=1, groups=channels))

    @staticmethod
    def ncc_loss(x, y):
        x = x - x.mean(dim=(-2, -1), keepdim=True)
        y = y - y.mean(dim=(-2, -1), keepdim=True)
        numerator = (x * y).mean(dim=(-2, -1))
        denominator = torch.sqrt((x * x).mean(dim=(-2, -1)) * (y * y).mean(dim=(-2, -1)) + 1e-06)
        ncc = numerator / denominator
        return 1.0 - ncc.mean()

    def information_preservation_loss(self, fused, vis, ir):
        pool_size = self.cfg['info_pool_size']
        fused_std = self.local_std(fused, pool_size)
        vis_std = self.local_std(vis, pool_size)
        ir_std = self.local_std(ir, pool_size)
        if self.cfg['use_local_info_guidance']:
            vis_edge = sobel_edges(vis)
            ir_edge = sobel_edges(ir)
            ir_saliency = F.relu(ir - F.avg_pool2d(ir, pool_size, 1, pool_size // 2))
            vis_quality = self._normalize_map(vis_std + vis_edge)
            ir_quality = self._normalize_map(ir_std + ir_edge + ir_saliency)
            temperature = self.cfg['info_quality_temperature']
            weights = torch.softmax(torch.cat([vis_quality, ir_quality], dim=1) * temperature, dim=1)
            vis_weight = weights[:, 0:1]
            ir_weight = weights[:, 1:2]
            max_std = torch.max(vis_std, ir_std)
            weighted_std = vis_weight * vis_std + ir_weight * ir_std
            target_mix = self.cfg['info_target_mix']
            target_std = target_mix * max_std + (1.0 - target_mix) * weighted_std
            content_target = vis_weight * vis + ir_weight * ir
            content_loss = F.l1_loss(fused, content_target)
        else:
            target_std = torch.max(vis_std, ir_std)
            content_loss = fused.new_tensor(0.0)
        contrast_loss = F.l1_loss(fused_std, target_std)
        corr_loss = 0.5 * (self.ncc_loss(fused, vis) + self.ncc_loss(fused, ir))
        contrast_w = self.cfg['info_contrast_weight']
        corr_w = self.cfg['info_corr_weight']
        content_w = self.cfg['info_content_weight']
        return contrast_w * contrast_loss + corr_w * corr_loss + content_w * content_loss

    def multi_scale_contrast_loss(self, fused, vis, ir):
        pool_sizes = self.cfg['contrast_pool_sizes']
        target_boost = self.cfg['contrast_target_boost']
        loss = fused.new_tensor(0.0)
        for pool_size in pool_sizes:
            fused_std = self.local_std(fused, pool_size)
            target_std = torch.max(self.local_std(vis, pool_size), self.local_std(ir, pool_size))
            target_std = torch.clamp(target_std * target_boost, 0.0, 1.0)
            loss = loss + F.l1_loss(fused_std, target_std)
        loss = loss / max(len(pool_sizes), 1)
        global_weight = self.cfg['contrast_global_weight']
        if global_weight > 0:
            fused_global = self._global_std(fused)
            target_global = torch.max(self._global_std(vis), self._global_std(ir))
            target_global = torch.clamp(target_global * target_boost, 0.0, 1.0)
            loss = loss + global_weight * F.l1_loss(fused_global, target_global)
        return loss

    def frequency_preservation_loss(self, fused, vis, ir):
        fused_lap = self.laplacian_response(fused)
        target_lap = torch.max(self.laplacian_response(vis), self.laplacian_response(ir))
        return F.l1_loss(fused_lap, target_lap)

    def low_frequency_distribution_loss(self, fused, vis, ir):
        pool_sizes = self.cfg['lowfreq_pool_sizes']
        temperature = self.cfg['lowfreq_quality_temperature']
        loss = fused.new_tensor(0.0)
        for pool_size in pool_sizes:
            padding = pool_size // 2
            fused_low = F.avg_pool2d(fused, pool_size, stride=1, padding=padding)
            vis_low = F.avg_pool2d(vis, pool_size, stride=1, padding=padding)
            ir_low = F.avg_pool2d(ir, pool_size, stride=1, padding=padding)
            vis_contrast = self.local_std(vis, pool_size)
            ir_contrast = self.local_std(ir, pool_size)
            ir_saliency = F.relu(ir_low - F.avg_pool2d(ir_low, 7, stride=1, padding=3))
            vis_quality = self._normalize_map(vis_contrast)
            ir_quality = self._normalize_map(ir_contrast + ir_saliency)
            weights = torch.softmax(torch.cat([vis_quality, ir_quality], dim=1) * temperature, dim=1)
            target_low = weights[:, 0:1] * vis_low + weights[:, 1:2] * ir_low
            loss = loss + F.l1_loss(fused_low, target_low)
        return loss / max(len(pool_sizes), 1)

    def entropy_spread_loss(self, fused, vis, ir):
        target_boost = self.cfg['entropy_target_boost']
        margin = self.cfg['entropy_margin']
        fused_std = self._global_std(fused)
        target_std = torch.max(self._global_std(vis), self._global_std(ir))
        target_std = target_boost * target_std
        std_loss = F.relu(target_std - fused_std + margin).mean()
        fused_mad = self._global_mad(fused)
        target_mad = torch.max(self._global_mad(vis), self._global_mad(ir))
        target_mad = target_boost * target_mad
        mad_loss = F.relu(target_mad - fused_mad + margin).mean()
        local_sizes = self.cfg['entropy_local_pool_sizes']
        local_loss = fused.new_tensor(0.0)
        for pool_size in local_sizes:
            fused_local = self.local_std(fused, pool_size)
            target_local = torch.max(self.local_std(vis, pool_size), self.local_std(ir, pool_size))
            target_local = target_boost * target_local
            local_loss = local_loss + F.relu(target_local - fused_local + margin).mean()
        local_loss = local_loss / max(len(local_sizes), 1)
        mad_weight = self.cfg['entropy_mad_weight']
        local_weight = self.cfg['entropy_local_weight']
        return std_loss + mad_weight * mad_loss + local_weight * local_loss

    def forward(self, fused, vis, ir, sem_feats=None, moe_aux=None, unc=None):
        loss_struct = F.l1_loss(fused, torch.max(vis, ir))
        loss_grad = gradient_loss(fused, vis, ir)
        loss_sobel = sobel_edge_loss(fused, vis, ir)
        loss_ssim = ssim_loss(fused, vis) + ssim_loss(fused, ir)
        loss_info = self.information_preservation_loss(fused, vis, ir)
        loss_contrast = self.multi_scale_contrast_loss(fused, vis, ir)
        loss_frequency = self.frequency_preservation_loss(fused, vis, ir)
        loss_lowfreq = self.low_frequency_distribution_loss(fused, vis, ir)
        loss_entropy = self.entropy_spread_loss(fused, vis, ir)
        scores = self.quality_scores(vis, ir)
        lambda_struct = self.adaptive_lambda('lambda_struct', scores['struct'])
        lambda_grad = self.adaptive_lambda('lambda_grad', scores['grad'])
        lambda_sobel = self.adaptive_lambda('lambda_sobel', scores['sobel'])
        lambda_ssim = self.adaptive_lambda('lambda_ssim', scores['ssim'])
        loss = lambda_struct * loss_struct + lambda_grad * loss_grad + lambda_sobel * loss_sobel + lambda_ssim * loss_ssim + self.cfg['lambda_info'] * loss_info + self.cfg['lambda_contrast'] * loss_contrast + self.cfg['lambda_frequency'] * loss_frequency + self.cfg['lambda_lowfreq'] * loss_lowfreq + self.cfg['lambda_entropy'] * loss_entropy
        if moe_aux is not None:
            loss += self.cfg['lambda_moe'] * moe_aux
        if unc is not None:
            loss += self.cfg['lambda_unc'] * unc
        return loss
