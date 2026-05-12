import torch
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.core.complexity_mask import ComplexityMask

def test_complexity_mask():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mask_gen = ComplexityMask().to(device)
    
    # 1. Create Vacuum Tile (State 0)
    # Smooth gradient
    vacuum_tile = torch.linspace(0, 0.1, 128*128).view(1, 1, 128, 128).repeat(1, 3, 1, 1)
    
    # 2. Create Texture Tile (State 1)
    # Repeating sine wave
    x = torch.linspace(0, 100, 128)
    y = torch.linspace(0, 100, 128)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
    texture_tile = torch.sin(grid_x) + torch.cos(grid_y)
    texture_tile = texture_tile.view(1, 1, 128, 128).repeat(1, 3, 1, 1)
    
    # 3. Create Detail Tile (State 2)
    # Random high-frequency noise
    detail_tile = torch.randn(1, 3, 128, 128)
    
    batch = torch.cat([vacuum_tile, texture_tile, detail_tile], dim=0).to(device)
    
    mask = mask_gen(batch)
    
    # Calculate scores manually for debug print in test
    x_gray = batch.mean(dim=1, keepdim=True)
    fft = torch.fft.rfft2(x_gray, norm='ortho')
    power_spectrum = torch.abs(fft)
    power_flat = power_spectrum.view(batch.shape[0], -1)
    power_no_dc = power_flat[:, 1:]
    max_power = torch.max(power_no_dc, dim=1)[0]
    mean_power = torch.mean(power_no_dc, dim=1)
    periodicity_scores = max_power / (mean_power + 1e-8)
    
    print(f"Input batch shape: {batch.shape}")
    print(f"Periodicity Scores: {periodicity_scores.tolist()}")
    print(f"Output mask: \n{mask.flatten().tolist()}")
    
    # Assertions
    assert mask[0] == 0, f"Expected State 0 (Vacuum), got {mask[0]}"
    assert mask[1] == 1, f"Expected State 1 (Texture), got {mask[1]}"
    assert mask[2] == 2, f"Expected State 2 (Detail), got {mask[2]}"
    
    print("Complexity Mask Test Passed!")

if __name__ == "__main__":
    test_complexity_mask()
