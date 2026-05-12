import torch
import torch.nn as nn
import torch.nn.functional as F

class PolynomialSurfaceFitter(nn.Module):
    """
    The Math Wizard: Fits a 3rd-degree polynomial surface to image tiles.
    Captures smooth gradients using only 10 coefficients per color channel.
    """
    def __init__(self, tile_size=128, lambda_reg=1e-4):
        super().__init__()
        self.tile_size = tile_size
        self.lambda_reg = lambda_reg
        
        # Precompute the coordinate grid and design matrix A
        # We only do this once to save GPU cycles
        self.register_buffer("A", self._build_design_matrix())

    def _build_design_matrix(self):
        """Builds the matrix A for the equation Ac = z"""
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, self.tile_size),
            torch.linspace(-1, 1, self.tile_size),
            indexing='ij'
        )
        x = x.reshape(-1)
        y = y.reshape(-1)

        # 3rd-degree basis: [1, x, y, x², y², xy, x³, y³, x²y, xy²]
        A = torch.stack([
            torch.ones_like(x),          # 1
            x,                           # x
            y,                           # y
            x**2,                        # x^2
            y**2,                        # y^2
            x*y,                         # xy
            x**3,                        # x^3
            y**3,                        # y^3
            (x**2)*y,                    # x^2y
            x*(y**2)                     # xy^2
        ], dim=1) 
        return A # Shape: (16384, 10)

    def fit(self, tiles):
        """
        Fits coefficients for a batch of tiles.
        Args:
            tiles: Tensor of shape (B, 3, 128, 128)
        Returns:
            coeffs: Tensor of shape (B, 3, 10)
        """
        B, C, H, W = tiles.shape
        z = tiles.reshape(B * C, -1).transpose(0, 1) # (16384, B*C)

        # Apply Tikhonov Regularization: (AᵀA + λI)⁻¹ Aᵀz
        # Using torch.linalg.lstsq is more numerically stable
        # We solve: A * coeffs = z
        # result.solution shape: (10, B*C)
        solution = torch.linalg.lstsq(self.A, z, rcond=self.lambda_reg).solution
        
        coeffs = solution.transpose(0, 1).reshape(B, C, 10)
        return coeffs

    def render(self, coeffs):
        """
        Regenerates pixels from coefficients.
        Args:
            coeffs: Tensor of shape (B, 3, 10)
        Returns:
            tiles: Tensor of shape (B, 3, 128, 128)
        """
        B, C, N = coeffs.shape
        # Flatten coeffs for matrix multiplication
        c = coeffs.reshape(B * C, N).transpose(0, 1) # (10, B*C)
        
        # Matrix multiply: A (16384, 10) @ c (10, B*C) -> (16384, B*C)
        z_hat = torch.matmul(self.A, c)
        
        tiles_hat = z_hat.transpose(0, 1).reshape(B, C, self.tile_size, self.tile_size)
        return tiles_hat

    def quantize_coeffs(self, coeffs, scale=127):
        """Prepares coefficients for .padox binary packing"""
        return torch.round(coeffs * scale).to(torch.int8)

    def dequantize_coeffs(self, q_coeffs, scale=127):
        """Restores coefficients from .padox binary data"""
        return q_coeffs.to(torch.float32) / scale
