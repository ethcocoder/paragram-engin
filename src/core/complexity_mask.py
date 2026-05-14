"""
complexity_mask.py — Aether-Blueprint v3.0
==========================================
Vectorized Fourier-Variance tile classifier.
 
Responsibility:
    Analyse every tile in a batch and assign one of three routing states:
        State 0 (GEOMETRIC)  — low entropy, smooth gradients → Polynomial engine
        State 1 (STRUCTURAL) — high periodicity, repeating textures → Codebook engine
        State 2 (NEURAL)     — high chaotic energy, faces/detail → Neural engine
 
    The decision uses two orthogonal signals computed in a single vectorised pass:
        • Spatial variance  : overall energy level of the tile
        • Fourier periodicity: ratio of spectral peak power to mean power
                               (high ratio → periodic/textured)
 
Design notes:
    - All operations are batched on the input device (CPU or CUDA).
    - Thresholds are configurable so the caller can tune routing ratios.
    - Returns both the integer mask and boolean index tensors for direct
      use by the orchestrator.
"""
 
import torch
import torch.nn as nn
 
 
class ComplexityMask(nn.Module):
    """
    Classifies image tiles into three routing states using Fourier-Variance analysis.
 
    Args:
        var_threshold   (float): Tiles with spatial variance below this value are
                                 routed to the Geometric engine (State 0).
                                 Default 0.002 works well for [0,1]-normalised tiles.
        fourier_ratio   (float): Tiles whose FFT peak/mean ratio exceeds this are
                                 routed to the Structural engine (State 1).
                                 Default 8.0 captures clear texture periodicity.
    """
 
    STATE_GEOMETRIC  = 0
    STATE_STRUCTURAL = 1
    STATE_NEURAL     = 2
 
    def __init__(self, var_threshold: float = 0.002, fourier_ratio: float = 8.0):
        super().__init__()
        self.var_threshold = var_threshold
        self.fourier_ratio = fourier_ratio
 
    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
 
    @staticmethod
    def _spatial_variance(tiles: torch.Tensor) -> torch.Tensor:
        """
        Compute per-tile spatial variance across H×W×C.
 
        Args:
            tiles: (B, C, H, W) float tensor in [0, 1].
        Returns:
            (B,) variance tensor.
        """
        B = tiles.shape[0]
        flat = tiles.view(B, -1)                    # (B, C*H*W)
        return flat.var(dim=1)                      # (B,)
 
    @staticmethod
    def _fourier_periodicity(tiles: torch.Tensor) -> torch.Tensor:
        """
        Compute per-tile FFT peak-to-mean ratio as a periodicity score.
 
        Args:
            tiles: (B, C, H, W) float tensor.
        Returns:
            (B,) periodicity ratio tensor.
        """
        # Greyscale approximation — fast and sufficient for routing
        grey = tiles.mean(dim=1)                    # (B, H, W)
 
        fft  = torch.fft.rfft2(grey)               # (B, H, W//2+1) complex
        mag  = fft.abs()                            # magnitude spectrum
 
        B    = mag.shape[0]
        flat = mag.view(B, -1)                      # (B, N)
 
        # Zero out DC component (index 0) before measuring peaks
        flat_no_dc          = flat.clone()
        flat_no_dc[:, 0]    = 0.0
 
        peak  = flat_no_dc.max(dim=1).values        # (B,)
        mean  = flat_no_dc.mean(dim=1).clamp(min=1e-6)
 
        return peak / mean                          # (B,) ratio
 
    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
 
    def forward(self, tiles: torch.Tensor) -> dict:
        """
        Classify a batch of tiles and return routing information.
 
        Args:
            tiles: (B, C, H, W) float tensor, values in [0, 1].
 
        Returns:
            dict with keys:
                'mask'       : (B,) int tensor — state per tile (0, 1, or 2)
                'geo_idx'    : 1-D LongTensor — batch indices routed to Geometric
                'struct_idx' : 1-D LongTensor — batch indices routed to Structural
                'neural_idx' : 1-D LongTensor — batch indices routed to Neural
                'stats'      : dict with float ratios for logging
        """
        with torch.no_grad():
            variance    = self._spatial_variance(tiles)       # (B,)
            periodicity = self._fourier_periodicity(tiles)    # (B,)
 
        B    = tiles.shape[0]
        mask = torch.full((B,), self.STATE_NEURAL,
                          dtype=torch.long, device=tiles.device)
 
        # State 0: low variance → Geometric (smooth gradient regions)
        is_geo               = variance < self.var_threshold
        mask[is_geo]         = self.STATE_GEOMETRIC
 
        # State 1: high periodicity AND not already geometric → Structural
        is_struct            = (~is_geo) & (periodicity > self.fourier_ratio)
        mask[is_struct]      = self.STATE_STRUCTURAL
 
        # State 2: everything else → Neural (default)
 
        geo_idx    = (mask == self.STATE_GEOMETRIC).nonzero(as_tuple=True)[0]
        struct_idx = (mask == self.STATE_STRUCTURAL).nonzero(as_tuple=True)[0]
        neural_idx = (mask == self.STATE_NEURAL).nonzero(as_tuple=True)[0]
 
        stats = {
            'pct_geometric'  : geo_idx.numel()    / B * 100,
            'pct_structural' : struct_idx.numel() / B * 100,
            'pct_neural'     : neural_idx.numel() / B * 100,
        }
 
        return {
            'mask'       : mask,
            'geo_idx'    : geo_idx,
            'struct_idx' : struct_idx,
            'neural_idx' : neural_idx,
            'stats'      : stats,
        }
 