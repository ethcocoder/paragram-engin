import torch
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.core.orchestrator import AetherOrchestrator

def test_full_integration():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    orchestrator = AetherOrchestrator().to(device)
    
    # 1. Prepare Hybrid Batch
    # Tile 0: Smooth Gradient (Geometric)
    y, x = torch.meshgrid(torch.linspace(-1,1,128), torch.linspace(-1,1,128), indexing='ij')
    vacuum = (0.5 + 0.1*x).view(1, 1, 128, 128).repeat(1, 3, 1, 1)
    
    # Tile 1: Repeating Pattern (Structural)
    # diagonal pattern
    pattern = torch.eye(128).view(1, 1, 128, 128).repeat(1, 3, 1, 1)
    
    # Tile 2: Complex Noise (Neural)
    detail = torch.randn(1, 3, 128, 128)
    
    batch = torch.cat([vacuum, pattern, detail], dim=0).to(device)
    B = batch.shape[0]
    
    # 2. Forward Pass (Encoding/Routing)
    print("Running Hybrid Forward Pass...")
    results = orchestrator(batch)
    
    mask = results['mask'].flatten().tolist()
    print(f"Orchestrator Mask: {mask}")
    
    # 3. Reconstruct
    print("Running Hybrid Reconstruction...")
    reconstructed = orchestrator.reconstruct(results, B)
    
    print(f"Reconstructed Shape: {reconstructed.shape}")
    
    assert reconstructed.shape == batch.shape
    
    # Check fidelity roughly
    for i in range(B):
        mse = torch.mean((batch[i] - reconstructed[i])**2).item()
        print(f"Tile {i} (State {mask[i]}) MSE: {mse:.2e}")
        
    print("Integration Test Passed!")

if __name__ == "__main__":
    test_full_integration()
