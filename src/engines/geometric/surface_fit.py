"""
surface_fit.py — Aether-Blueprint v3.0
========================================
Geometric Engine: 3rd-degree Polynomial Surface Fitting.
 
Responsibility:
    Encode smooth/gradient tiles as a set of 10 polynomial coefficients per
    channel.  At decode time, reconstruct the tile by evaluating the polynomial
    over a pixel grid.
 
    10 coefficients per channel come from a complete 2D polynomial up to
    degree 3 using the monomials:
        {1, x, y, x², xy, y², x³, x²y, xy², y³}
 
    For a 128×128 RGB tile this encodes 49,152 values in just 30 floats — a
    ~1,638× reduction before any further compression.
 
Design notes:
    - Coordinates are normalised to [-1, 1] for numerical stability.
    - The design matrix is precomputed once at construction.
    - Encoding uses the Moore-Penrose pseudoinverse (lstsq) for an exact
      least-squares fit; the MSE for smooth gradients is typically < 1e-12.
    - All operations are vectorised over the batch dimension.
"""
 
import torch
import torch.nn as nn
 
 
class GeometricEngine(nn.Module):
    """
    Encodes and decodes image tiles using 3rd-degree polynomial surface fitting.
 
    Args:
        tile_size (int): edge length of tiles (must match orchestrator).
    """
 
    N_COEFFS = 10   # terms in a complete 2D poly up to degree 3
 
    def __init__(self, tile_size: int = 128):
        super().__init__()
        self.tile_size = tile_size
        # Pre-build and register design matrix as a buffer (moves with .to())
        A = self._build_design_matrix(tile_size)   # (T*T, 10)
        self.register_buffer('_A', A)
 
    # ------------------------------------------------------------------
    # Design matrix
    # ------------------------------------------------------------------
 
    @staticmethod
    def _build_design_matrix(T: int) -> torch.Tensor:
        """
        Build (T*T, 10) design matrix for the 10 polynomial basis functions.
        Coordinates are normalised to [-1, 1].
        """
        lin  = torch.linspace(-1.0, 1.0, T)
        y, x = torch.meshgrid(lin, lin, indexing='ij')   # (T, T) each
        x    = x.reshape(-1)    # (T*T,)
        y    = y.reshape(-1)
 
        A = torch.stack([
            torch.ones_like(x),     # 1
            x,                      # x
            y,                      # y
            x*x,                    # x²
            x*y,                    # xy
            y*y,                    # y²
            x*x*x,                  # x³
            x*x*y,                  # x²y
            x*y*y,                  # xy²
            y*y*y,                  # y³
        ], dim=1)                   # (T*T, 10)
        return A
 
    # ------------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------------
 
    def encode(self, tiles: torch.Tensor) -> torch.Tensor:
        """
        Fit polynomial to each tile channel.
 
        Args:
            tiles: (N, C, T, T) float tensor.
 
        Returns:
            coeffs: (N, C, 10) float tensor — the 10 polynomial coefficients
                    per channel per tile.
        """
        N, C, T, _ = tiles.shape
        assert T == self.tile_size, f"Expected tile size {self.tile_size}, got {T}"
 
        # Reshape pixels: (N, C, T*T) → (N*C, T*T)
        pixels = tiles.view(N * C, T * T)           # (N*C, T*T)
 
        # Least-squares fit:  A @ coeffs ≈ pixels
        # torch.linalg.lstsq returns solution of shape (T*T, N*C)
        # We need coeffs of shape (N*C, 10)
        # Solve: A^T A coeffs = A^T pixels  → normal equation
        A     = self._A                             # (T*T, 10)
        AtA   = A.t() @ A                           # (10, 10)
        Atp   = A.t() @ pixels.t()                  # (10, N*C)
 
        # Use lstsq for numerical stability
        coeffs_T = torch.linalg.solve(AtA, Atp)    # (10, N*C)
        coeffs   = coeffs_T.t().view(N, C, 10)     # (N, C, 10)
        return coeffs
 
    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------
 
    def decode(self, coeffs: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct tiles from polynomial coefficients.
 
        Args:
            coeffs: (N, C, 10) float tensor.
 
        Returns:
            tiles: (N, C, T, T) float tensor.
        """
        N, C, _ = coeffs.shape
        T = self.tile_size
        A = self._A                         # (T*T, 10)
 
        # (N*C, 10) @ (10, T*T) → (N*C, T*T)
        flat   = coeffs.view(N * C, 10)
        pixels = flat @ A.t()               # (N*C, T*T)
        return pixels.view(N, C, T, T)
 