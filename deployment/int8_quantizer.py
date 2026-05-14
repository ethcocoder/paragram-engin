"""
int8_quantizer.py — Aether-Blueprint v3.0
=========================================
Post-Training Quantization (PTQ) Pipeline.

Responsibility:
    Convert the trained PyTorch floating-point model (NeuralEngine) into an
    INT8 quantized PyTorch model. While mobile_export.py handles ONNX quantization,
    this script creates native PyTorch quantized models for direct use in
    PyTorch Mobile or TorchScript.
"""

import os
import sys
import torch
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.engines.neural.lightweight import LightweightNeuralEngine

def calibrate_and_quantize(checkpoint_path: str, output_path: str, tile_size: int = 128, latent_dim: int = 128):
    """
    Performs dynamic quantization on the Neural Engine.
    Dynamic quantization is weight-only for linear/LSTM layers by default in PyTorch,
    but we can extend this to full static quantization if calibration data is provided.
    """
    print(f"🚀 PyTorch Native INT8 Quantizer")
    print(f"   Input : {checkpoint_path}")
    print(f"   Output: {output_path}")
    
    device = torch.device('cpu') # Quantization is best done on CPU
    neural_engine = LightweightNeuralEngine(tile_size=tile_size, latent_dim=latent_dim).to(device)
    neural_engine.eval()
    
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
        # Extract neural engine weights
        neural_keys = {k.replace('neural_engine.', ''): v 
                       for k, v in state_dict.items() if 'neural_engine' in k}
        if neural_keys:
            neural_engine.load_state_dict(neural_keys, strict=False)
            print("✅ Loaded Neural Engine weights.")
        else:
            print("⚠️ No neural engine weights found. Quantizing random weights.")
    else:
        print(f"⚠️ Checkpoint {checkpoint_path} not found. Quantizing random weights.")

    # Apply PyTorch Dynamic Quantization
    # Note: torch.quantization.quantize_dynamic targets nn.Linear, nn.LSTM, etc.
    # For Conv2d, static quantization or QAT is preferred.
    print("⏳ Applying Dynamic INT8 Quantization...")
    quantized_engine = torch.ao.quantization.quantize_dynamic(
        neural_engine, 
        {torch.nn.Linear, torch.nn.Conv2d}, # In PyTorch 1.13+, Conv2d dynamic quant is partially supported
        dtype=torch.qint8
    )
    
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    torch.save(quantized_engine.state_dict(), output_path)
    print(f"✅ Quantized model state dict saved to: {output_path}")

    # File size comparison
    orig_size = os.path.getsize(checkpoint_path) if os.path.exists(checkpoint_path) else 1
    new_size = os.path.getsize(output_path)
    print(f"   Size reduction: {new_size/1024/1024:.2f} MB vs {orig_size/1024/1024:.2f} MB")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PyTorch INT8 Quantizer")
    parser.add_argument('--checkpoint', type=str, default='stage3_qat_final.pth')
    parser.add_argument('--output', type=str, default='deployment/exported/neural_engine_int8.pth')
    args = parser.parse_args()
    
    calibrate_and_quantize(args.checkpoint, args.output)
