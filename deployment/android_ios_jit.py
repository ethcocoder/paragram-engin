"""
android_ios_jit.py — Aether-Blueprint v3.0
==========================================
TorchScript and Just-In-Time (JIT) compiler optimizations.

Responsibility:
    Export the NeuralEngine and GeometricEngine using torch.jit.trace.
    These optimized .pt files can be deployed directly via PyTorch Mobile
    on iOS and Android devices, natively leveraging mobile NPUs/GPUs
    via Metal or Vulkan backends.
"""

import os
import sys
import torch
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.engines.geometric.surface_fit import GeometricEngine
from src.engines.neural.lightweight import LightweightNeuralEngine

# Wrappers for Tracing
class NeuralEncoderJIT(torch.nn.Module):
    def __init__(self, engine):
        super().__init__()
        self.engine = engine
    def forward(self, tiles: torch.Tensor) -> torch.Tensor:
        return self.engine.encode(tiles)

class NeuralDecoderJIT(torch.nn.Module):
    def __init__(self, engine):
        super().__init__()
        self.engine = engine
    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.engine.decode(latents)

def compile_torchscript(checkpoint_path: str, output_dir: str, tile_size: int = 128, latent_dim: int = 128):
    """
    Exports engines as TorchScript.
    """
    print(f"🚀 Aether-Blueprint JIT Compiler (PyTorch Mobile)")
    device = torch.device('cpu')
    os.makedirs(output_dir, exist_ok=True)
    
    neural_engine = LightweightNeuralEngine(tile_size=tile_size, latent_dim=latent_dim).to(device)
    neural_engine.eval()
    
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
        neural_keys = {k.replace('neural_engine.', ''): v for k, v in state_dict.items() if 'neural_engine' in k}
        if neural_keys:
            neural_engine.load_state_dict(neural_keys, strict=False)
            print("✅ Loaded Neural Engine weights.")
    
    # ── JIT Trace Neural Engine ──
    print("\n⏳ Tracing Neural Encoder...")
    encoder_jit = NeuralEncoderJIT(neural_engine)
    dummy_tile = torch.randn(1, 3, tile_size, tile_size)
    traced_encoder = torch.jit.trace(encoder_jit, dummy_tile)
    traced_encoder.save(os.path.join(output_dir, "neural_encoder.pt"))
    print("✅ Saved neural_encoder.pt")
    
    print("⏳ Tracing Neural Decoder...")
    decoder_jit = NeuralDecoderJIT(neural_engine)
    dummy_latent = torch.randn(1, latent_dim)
    traced_decoder = torch.jit.trace(decoder_jit, dummy_latent)
    traced_decoder.save(os.path.join(output_dir, "neural_decoder.pt"))
    print("✅ Saved neural_decoder.pt")

    # ── Mobile optimization pass ──
    try:
        from torch.utils.mobile_optimizer import optimize_for_mobile
        print("\n⏳ Applying optimize_for_mobile()...")
        opt_encoder = optimize_for_mobile(traced_encoder)
        opt_decoder = optimize_for_mobile(traced_decoder)
        
        opt_encoder._save_for_lite_interpreter(os.path.join(output_dir, "neural_encoder.ptl"))
        opt_decoder._save_for_lite_interpreter(os.path.join(output_dir, "neural_decoder.ptl"))
        print("✅ Saved .ptl files for PyTorch Mobile Lite Interpreter.")
    except ImportError:
        print("⚠️ optimize_for_mobile not available in this PyTorch version.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="TorchScript JIT Compiler")
    parser.add_argument('--checkpoint', type=str, default='stage3_qat_final.pth')
    parser.add_argument('--output_dir', type=str, default='deployment/mobile')
    args = parser.parse_args()
    
    compile_torchscript(args.checkpoint, args.output_dir)
