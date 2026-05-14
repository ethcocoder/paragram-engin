"""
gradient_gen.py — Aether-Blueprint v3.0
=========================================
Geometric Engine Extension: Parametric Gradient Reconstruction.
 
Responsibility:
    Generate smooth colour/luminance gradients from a compact parametric
    description.  Used by the Geometric Engine for tiles that are purely
    transitional (sky-to-horizon, vignettes, lens blur backgrounds).
 
    A gradient tile is described by:
        - angle      : direction in degrees [0, 360)
        - stop_colors: list of (position ∈ [0,1], RGB tuple) stop points
                       (at least 2 stops required)
 
    This produces a far more compact representation than even polynomial
    coefficients for tiles that are visually linear gradients.
 
Design notes:
    - Parameterises a 1-D projection of the tile at a given angle.
    - Interpolates stop colours using linear (Lerp) or smooth-step mode.
    - Fully differentiable for training integration.
"""
 
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
 
 
class GradientGenerator(nn.Module):
    """
    Generates a tile from a parametric gradient description.
 
    Args:
        tile_size (int): edge length of output tiles.
    """
 
    def __init__(self, tile_size: int = 128):
        super().__init__()
        self.tile_size = tile_size
        # Pre-build normalised coordinate grids: (T, T)
        lin  = torch.linspace(0.0, 1.0, tile_size)
        y_g, x_g = torch.meshgrid(lin, lin, indexing='ij')
        self.register_buffer('_x_grid', x_g)
        self.register_buffer('_y_grid', y_g)
 
    def _projection(self, angle_deg: float) -> torch.Tensor:
        """
        Project the 2-D pixel grid onto a 1-D axis at the given angle.
        Returns a (T, T) tensor of values in [0, 1].
        """
        rad = math.radians(angle_deg)
        dx  = math.cos(rad)
        dy  = math.sin(rad)
        proj = dx * self._x_grid + dy * self._y_grid   # (T, T)
        # Normalise to [0, 1]
        proj = (proj - proj.min()) / (proj.max() - proj.min() + 1e-8)
        return proj
 
    def generate(self,
                 angle_deg: float,
                 stop_positions: list,
                 stop_colors: list,
                 smooth: bool = True) -> torch.Tensor:
        """
        Generate a single gradient tile.
 
        Args:
            angle_deg      : gradient direction in degrees
            stop_positions : list of N floats in [0, 1], sorted ascending
            stop_colors    : list of N (R, G, B) tuples, values in [0, 1]
            smooth         : if True use smooth-step interpolation
 
        Returns:
            (1, 3, T, T) float tensor in [0, 1]
        """
        assert len(stop_positions) == len(stop_colors) >= 2
 
        T    = self.tile_size
        proj = self._projection(angle_deg)          # (T, T)
 
        if smooth:
            # Smooth-step: 3t² - 2t³
            proj = proj * proj * (3.0 - 2.0 * proj)
 
        # Build output tensor
        out = torch.zeros(3, T, T, device=proj.device)
 
        for i in range(len(stop_positions) - 1):
            p0, p1 = stop_positions[i], stop_positions[i + 1]
            c0 = torch.tensor(stop_colors[i],     device=proj.device)  # (3,)
            c1 = torch.tensor(stop_colors[i + 1], device=proj.device)
 
            # Mask for this segment
            in_seg = (proj >= p0) & (proj <= p1)
            t_seg  = ((proj - p0) / max(p1 - p0, 1e-8)).clamp(0.0, 1.0)
 
            for ch in range(3):
                lerped = c0[ch] + t_seg * (c1[ch] - c0[ch])
                out[ch] = torch.where(in_seg, lerped, out[ch])
 
        return out.unsqueeze(0)     # (1, 3, T, T)
 
    def encode_tile(self, tile: torch.Tensor, n_stops: int = 3) -> dict:
        """
        Fit a gradient description to a tile using a simple angle-sweep search.
 
        Args:
            tile   : (1, 3, T, T) float tensor
            n_stops: number of gradient stops to fit
 
        Returns:
            dict with 'angle', 'positions', 'colors'
        """
        T = self.tile_size
        best_mse   = float('inf')
        best_angle = 0.0
 
        # Coarse angle sweep (every 15°)
        for angle in range(0, 180, 15):
            proj  = self._projection(float(angle)).reshape(-1)   # (T*T,)
            order = proj.argsort()
            # Sample mean color at n_stops positions along projection
            chunk = T * T // n_stops
            colors, positions = [], []
            for k in range(n_stops):
                idx = order[k * chunk : (k + 1) * chunk]
                flat_tile = tile[0].reshape(3, T * T)
                mean_c    = flat_tile[:, idx].mean(dim=1).tolist()
                colors.append(mean_c)
                positions.append(float(proj[idx].mean()))
 
            # Quick MSE check
            recon = self.generate(float(angle), positions, colors, smooth=True)
            mse   = F.mse_loss(recon, tile).item()
            if mse < best_mse:
                best_mse   = mse
                best_angle = float(angle)
                best_pos   = positions
                best_col   = colors
 
        return {
            'angle'    : best_angle,
            'positions': best_pos,
            'colors'   : best_col,
            'mse'      : best_mse,
        }
 
    def forward(self, params_batch: list) -> torch.Tensor:
        """
        Batch generate from a list of parameter dicts.
 
        Args:
            params_batch: list of dicts, each with 'angle', 'positions', 'colors'
 
        Returns:
            (N, 3, T, T) tensor
        """
        tiles = [self.generate(p['angle'], p['positions'], p['colors'])
                 for p in params_batch]
        return torch.cat(tiles, dim=0)