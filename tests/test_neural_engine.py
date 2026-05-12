import torch
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.engines.neural.lightweight import DetailEngine

def test_neural_engine():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DetailEngine().to(device)
    
    # 1. Parameter Count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params / 1e6:.2f}M")
    assert total_params < 2e6, f"Too many parameters: {total_params}"
    
    # 2. Forward Pass
    x = torch.randn(1, 3, 128, 128).to(device)
    
    # Test Mixed Precision
    with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
        latent = model.encode(x)
        out = model.decode(latent)
        
    print(f"Input shape: {x.shape}")
    print(f"Latent shape: {latent.shape}")
    print(f"Output shape: {out.shape}")
    
    assert latent.shape == (1, 192, 16, 16), f"Wrong latent shape: {latent.shape}"
    assert out.shape == (1, 3, 128, 128), f"Wrong output shape: {out.shape}"
    
    # 3. Test Gradient Flow (basic check)
    out.mean().backward()
    print("Gradient check passed.")
    
    print("Neural Engine Test Passed!")

if __name__ == "__main__":
    test_neural_engine()
