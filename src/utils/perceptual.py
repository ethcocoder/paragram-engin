"""
perceptual.py — Aether-Blueprint v3.0
=======================================
Perceptual Loss: LPIPS + MS-SSIM + Human Eye Tuning.

Responsibility:
    Provide differentiable perceptual loss functions that measure visual
    similarity the way human eyes do — not pixel-level MSE.

    Two components:
        1. LPIPS (Learned Perceptual Image Patch Similarity)
           Uses VGG16 feature differences.  Penalises blurring and
           texture washing that MSE misses entirely.

        2. MS-SSIM (Multi-Scale Structural Similarity)
           Compares luminance, contrast, and structure at 5 scales.
           Handles compression artifacts and blocking.

    Combined loss:
        L_perceptual = α * LPIPS + β * (1 - MS_SSIM) + γ * MSE

    This combined loss is the gate to Stage 2 training.  Without it,
    the model optimises for numerical accuracy but produces washed-out,
    slightly blurry reconstructions that MSE cannot penalise.

Usage:
    criterion = PerceptualLoss(lpips_weight=0.5, ssim_weight=0.4, mse_weight=0.1)
    loss = criterion(recon, target)
    loss.backward()

Notes:
    - LPIPS requires the lpips package (pip install lpips).
    - Falls back to VGG-feature L2 if lpips is not installed.
    - All inputs should be (B, 3, H, W) float tensors in [0, 1].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# MS-SSIM
# ──────────────────────────────────────────────────────────────────────────────

_MS_SSIM_WEIGHTS = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])


def _gaussian_kernel(kernel_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """1-D Gaussian kernel, returns (kernel_size,)."""
    x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    g = torch.exp(-x**2 / (2 * sigma**2))
    return g / g.sum()


def _ssim_single_scale(img1: torch.Tensor,
                        img2: torch.Tensor,
                        kernel: torch.Tensor,
                        C1: float = 0.01**2,
                        C2: float = 0.03**2) -> torch.Tensor:
    """
    Compute SSIM map for a single scale.

    Args:
        img1, img2 : (B, 1, H, W) — single-channel
        kernel     : (1, 1, K, K) Gaussian kernel
        C1, C2     : stability constants

    Returns:
        (B,) mean SSIM per image
    """
    mu1    = F.conv2d(img1, kernel, padding=kernel.shape[-1]//2, groups=1)
    mu2    = F.conv2d(img2, kernel, padding=kernel.shape[-1]//2, groups=1)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu12   = mu1 * mu2

    sig1_sq = F.conv2d(img1 * img1, kernel, padding=kernel.shape[-1]//2) - mu1_sq
    sig2_sq = F.conv2d(img2 * img2, kernel, padding=kernel.shape[-1]//2) - mu2_sq
    sig12   = F.conv2d(img1 * img2, kernel, padding=kernel.shape[-1]//2) - mu12

    numerator   = (2 * mu12   + C1) * (2 * sig12   + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sig1_sq + sig2_sq + C2)
    ssim_map    = numerator / denominator.clamp(min=1e-8)
    return ssim_map.mean(dim=[1, 2, 3])   # (B,)


class MSSSIM(nn.Module):
    """
    Multi-Scale SSIM loss.

    Args:
        kernel_size (int): Gaussian filter size.
        sigma       (float): Gaussian sigma.
        n_scales    (int): number of scales (default 5, max for 128px tiles).
    """

    def __init__(self, kernel_size: int = 11, sigma: float = 1.5, n_scales: int = 5):
        super().__init__()
        self.n_scales = n_scales
        k1d  = _gaussian_kernel(kernel_size, sigma)
        k2d  = k1d.unsqueeze(0) * k1d.unsqueeze(1)         # (K, K)
        kernel = k2d.unsqueeze(0).unsqueeze(0)              # (1, 1, K, K)
        self.register_buffer('kernel', kernel)
        weights = _MS_SSIM_WEIGHTS[:n_scales]
        self.register_buffer('weights', weights / weights.sum())

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute MS-SSIM.

        Args:
            pred, target: (B, C, H, W) in [0, 1]
        Returns:
            scalar MS-SSIM value in [0, 1] (higher = more similar)
        """
        # Convert to luminance (greyscale) for SSIM computation
        weights_rgb = pred.new_tensor([0.2989, 0.5870, 0.1140])
        p_lum = (pred   * weights_rgb.view(1, 3, 1, 1)).sum(1, keepdim=True)
        t_lum = (target * weights_rgb.view(1, 3, 1, 1)).sum(1, keepdim=True)

        mcs_list   = []
        ssim_final = None
        p, t       = p_lum, t_lum

        for scale in range(self.n_scales):
            ssim_val = _ssim_single_scale(p, t, self.kernel)

            if scale < self.n_scales - 1:
                # Contrast-structure only for intermediate scales
                mcs_list.append(ssim_val)
                p = F.avg_pool2d(p, kernel_size=2, stride=2)
                t = F.avg_pool2d(t, kernel_size=2, stride=2)
            else:
                ssim_final = ssim_val

        # Product of weighted contrast-structure terms × final SSIM
        result = ssim_final
        for i, mcs in enumerate(mcs_list):
            result = result * (mcs ** self.weights[i])

        return result.mean()    # scalar


# ──────────────────────────────────────────────────────────────────────────────
# LPIPS (with graceful fallback)
# ──────────────────────────────────────────────────────────────────────────────

class LPIPSLoss(nn.Module):
    """
    LPIPS wrapper that tries the `lpips` package and falls back to
    VGG-feature L2 if the package is not installed.
    """

    def __init__(self, net: str = 'vgg'):
        super().__init__()
        self._use_lpips = False
        try:
            import lpips as lpips_lib
            self._lpips = lpips_lib.LPIPS(net=net)
            self._use_lpips = True
        except ImportError:
            # Fallback: VGG feature extraction using torchvision
            try:
                import torchvision.models as tvm
                vgg = tvm.vgg16(weights=tvm.VGG16_Weights.IMAGENET1K_V1)
                self._vgg_features = nn.ModuleList(list(vgg.features[:23]))
                for p in self._vgg_features.parameters():
                    p.requires_grad_(False)
            except Exception:
                self._vgg_features = None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred, target : (B, 3, H, W) in [0, 1]
        Returns:
            scalar LPIPS distance (lower = more similar)
        """
        if self._use_lpips:
            # lpips expects inputs in [-1, 1]
            p = pred   * 2 - 1
            t = target * 2 - 1
            return self._lpips(p, t).mean()

        if self._vgg_features is not None:
            # VGG feature L2 fallback
            mean = pred.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std  = pred.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            p = (pred   - mean) / std
            t = (target - mean) / std
            loss = pred.new_tensor(0.0)
            for layer in self._vgg_features:
                p = layer(p)
                t = layer(t)
            loss = F.mse_loss(p, t)
            return loss

        # Last resort: simple L1 in image space
        return F.l1_loss(pred, target)


# ──────────────────────────────────────────────────────────────────────────────
# Combined PerceptualLoss
# ──────────────────────────────────────────────────────────────────────────────

class PerceptualLoss(nn.Module):
    """
    Combined perceptual loss for Stage 2 training.

        L = lpips_weight  * LPIPS(pred, target)
          + ssim_weight   * (1 - MS_SSIM(pred, target))
          + mse_weight    * MSE(pred, target)

    Args:
        lpips_weight (float): weight for LPIPS component.
        ssim_weight  (float): weight for MS-SSIM component.
        mse_weight   (float): weight for pixel MSE component.
    """

    def __init__(self,
                 lpips_weight: float = 0.5,
                 ssim_weight:  float = 0.4,
                 mse_weight:   float = 0.1):
        super().__init__()
        self.lpips_weight = lpips_weight
        self.ssim_weight  = ssim_weight
        self.mse_weight   = mse_weight

        self.lpips_loss = LPIPSLoss(net='vgg')
        self.msssim     = MSSSIM()

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> tuple:
        """
        Args:
            pred, target: (B, 3, H, W) in [0, 1]

        Returns:
            (total_loss, loss_dict)  — loss_dict for logging
        """
        lpips_val = self.lpips_loss(pred, target)
        ssim_val  = self.msssim(pred, target)
        mse_val   = F.mse_loss(pred, target)

        ssim_loss = 1.0 - ssim_val
        total     = (self.lpips_weight * lpips_val
                   + self.ssim_weight  * ssim_loss
                   + self.mse_weight   * mse_val)

        loss_dict = {
            'lpips'     : lpips_val.item(),
            'ms_ssim'   : ssim_val.item(),
            'ssim_loss' : ssim_loss.item(),
            'mse'       : mse_val.item(),
            'total'     : total.item(),
        }
        return total, loss_dict