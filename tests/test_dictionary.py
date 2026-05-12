import torch
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.engines.structural.dictionary import StructuralEngine

def test_structural_engine():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    engine = StructuralEngine(num_entries=512, patch_size=32).to(device)
    
    # 1. Setup a known pattern in the codebook at index 42
    # A diagonal line pattern
    pattern = torch.eye(32).view(1, 1, 32, 32).repeat(1, 3, 1, 1).to(device)
    with torch.no_grad():
        engine.codebook_module.codebook[42] = pattern.squeeze()
        
    # 2. Create an input tile (128x128) where one 32x32 patch is this pattern rotated 90 deg
    # Target: Patch at (0,0), rotated 1 time (90 deg), Gain=2.0, Bias=0.1
    input_tile = torch.zeros(1, 3, 128, 128).to(device)
    target_patch = torch.rot90(pattern, 1, [2, 3]) * 2.0 + 0.1
    input_tile[0, :, 0:32, 0:32] = target_patch
    
    # 3. Match
    indices, rotations, gains, biases, fallback_mask = engine(input_tile)
    
    print(f"Match result for patch [0,0]:")
    print(f"Index: {indices[0].item()}, Rotation: {rotations[0].item()}, Gain: {gains[0].item():.2f}, Bias: {biases[0].item():.2f}")
    
    # Assertions for the first patch
    assert indices[0] == 42, f"Expected index 42, got {indices[0]}"
    assert rotations[0] == 1, f"Expected rotation 1, got {rotations[0]}"
    assert abs(gains[0] - 2.0) < 1e-2, f"Expected gain 2.0, got {gains[0]}"
    assert abs(biases[0] - 0.1) < 1e-2, f"Expected bias 0.1, got {biases[0]}"
    
    # 4. Render
    reconstructed = engine.render(indices, rotations, gains, biases, B=1)
    
    # Check if the target patch was reconstructed correctly
    mse = torch.mean((input_tile[0, :, 0:32, 0:32] - reconstructed[0, :, 0:32, 0:32])**2)
    print(f"Patch reconstruction MSE: {mse.item():.2e}")
    assert mse < 1e-5, f"Reconstruction MSE too high: {mse.item()}"
    
    # 5. Test Fallback with random noise
    noise_tile = torch.randn(1, 3, 128, 128).to(device)
    _, _, _, _, fallback_mask_noise = engine(noise_tile)
    
    fallback_count = fallback_mask_noise.sum().item()
    print(f"Fallback patches for noise: {fallback_count}/16")
    assert fallback_count > 0, "Noise should trigger some fallback"
    
    print("Structural Engine Test Passed!")

if __name__ == "__main__":
    test_structural_engine()
