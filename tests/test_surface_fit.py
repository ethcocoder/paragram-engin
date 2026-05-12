import torch
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.engines.geometric.surface_fit import PolynomialSurfaceFitter

def test_surface_fit():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fitter = PolynomialSurfaceFitter().to(device)
    
    # 1. Create a synthetic smooth gradient (Vacuum mode style)
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, 128),
        torch.linspace(-1, 1, 128),
        indexing='ij'
    )
    # Target: z = 0.5 + 0.2x + 0.1y + 0.05x^2
    target_z = 0.5 + 0.2*x + 0.1*y + 0.05*(x**2)
    target_tile = target_z.view(1, 1, 128, 128).repeat(1, 3, 1, 1).to(device)
    
    # 2. Fit
    coeffs = fitter.fit(target_tile)
    print(f"Fitted coefficients shape: {coeffs.shape}")
    
    # 3. Render
    reconstructed = fitter.render(coeffs)
    
    # 4. Measure Fidelity
    mse = torch.mean((target_tile - reconstructed)**2)
    print(f"Reconstruction MSE: {mse.item():.2e}")
    
    assert mse < 1e-6, f"MSE too high: {mse.item()}"
    
    # 5. Test Quantization
    q_coeffs = fitter.quantize_coeffs(coeffs)
    dq_coeffs = fitter.dequantize_coeffs(q_coeffs)
    
    reconstructed_q = fitter.render(dq_coeffs)
    mse_q = torch.mean((target_tile - reconstructed_q)**2)
    print(f"Quantized Reconstruction MSE: {mse_q.item():.2e}")
    
    assert mse_q < 1e-3, f"Quantized MSE too high: {mse_q.item()}"
    
    print("Surface Fitter Test Passed!")

if __name__ == "__main__":
    test_surface_fit()
