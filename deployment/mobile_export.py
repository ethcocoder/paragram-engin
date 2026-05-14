"""
mobile_export.py — Aether-Blueprint v3.0
=========================================
Mobile Deployment Bridge.

Responsibility:
    Export trained Aether-Blueprint engines to mobile-ready formats:
        1. ONNX (opset 17) — universal interchange format.
        2. INT8 Quantized ONNX — via onnxruntime static quantization.

    Only the NeuralEngine and GeometricEngine are exported. The StructuralEngine
    is a pure lookup table (codebook + affine) that runs natively on any platform
    without needing a neural runtime.

Usage:
    python deployment/mobile_export.py \\
        --checkpoint stage2_perceptual.pth \\
        --output_dir deployment/exported \\
        --quantize

Notes:
    - ONNX opset 17 supports all ops used (Conv2d, PixelShuffle, LayerNorm, etc.)
    - INT8 quantization uses MinMax calibration (no calibration dataset needed
      for weight-only quantization; full static quantization requires a small
      calibration set).
    - Exported models can be loaded by ONNX Runtime, CoreML (via onnx-coreml),
      or TFLite (via onnx-tf).
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────────────────────
# Engine Wrappers (clean forward-only modules for ONNX tracing)
# ──────────────────────────────────────────────────────────────────────────────

class NeuralEncoderWrapper(nn.Module):
    """Wraps NeuralEngine.encode() for ONNX export."""
    def __init__(self, engine):
        super().__init__()
        self.engine = engine

    def forward(self, tiles: torch.Tensor) -> torch.Tensor:
        return self.engine.encode(tiles)


class NeuralDecoderWrapper(nn.Module):
    """Wraps NeuralEngine.decode() for ONNX export."""
    def __init__(self, engine):
        super().__init__()
        self.engine = engine

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.engine.decode(latents)


class GeometricEncoderWrapper(nn.Module):
    """Wraps GeometricEngine.encode() for ONNX export."""
    def __init__(self, engine):
        super().__init__()
        self.engine = engine

    def forward(self, tiles: torch.Tensor) -> torch.Tensor:
        return self.engine.encode(tiles)


class GeometricDecoderWrapper(nn.Module):
    """Wraps GeometricEngine.decode() for ONNX export."""
    def __init__(self, engine):
        super().__init__()
        self.engine = engine

    def forward(self, coeffs: torch.Tensor) -> torch.Tensor:
        return self.engine.decode(coeffs)


# ──────────────────────────────────────────────────────────────────────────────
# ONNX Export
# ──────────────────────────────────────────────────────────────────────────────

def export_to_onnx(model: nn.Module,
                   dummy_input: torch.Tensor,
                   path: str,
                   input_names: list = None,
                   output_names: list = None,
                   opset_version: int = 17,
                   dynamic_axes: dict = None) -> str:
    """
    Export a PyTorch module to ONNX format.

    Args:
        model        : nn.Module to export.
        dummy_input  : example input tensor for tracing.
        path         : output .onnx file path.
        input_names  : names for input tensors.
        output_names : names for output tensors.
        opset_version: ONNX opset (default 17).
        dynamic_axes : dict of dynamic axis specifications.

    Returns:
        Path to the saved .onnx file.
    """
    model.eval()
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)

    if input_names is None:
        input_names = ['input']
    if output_names is None:
        output_names = ['output']
    if dynamic_axes is None:
        dynamic_axes = {input_names[0]: {0: 'batch_size'},
                        output_names[0]: {0: 'batch_size'}}

    torch.onnx.export(
        model,
        dummy_input,
        path,
        opset_version=opset_version,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )

    # Verify the exported model
    try:
        import onnx
        onnx_model = onnx.load(path)
        onnx.checker.check_model(onnx_model)
        print(f"  ✅ ONNX export verified: {path}")
    except ImportError:
        print(f"  ⚠️ onnx package not installed — skipping verification.")
    except Exception as e:
        print(f"  ⚠️ ONNX verification warning: {e}")

    file_size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"     Size: {file_size_mb:.2f} MB")
    return path


def quantize_onnx_int8(input_path: str,
                       output_path: str,
                       calibration_data: list = None) -> str:
    """
    Apply INT8 quantization to an ONNX model.

    Uses onnxruntime.quantization for weight-only or static quantization.

    Args:
        input_path       : path to the float32 .onnx model.
        output_path      : path to save the quantized .onnx model.
        calibration_data : optional list of numpy arrays for static quantization.

    Returns:
        Path to the quantized model.
    """
    try:
        from onnxruntime.quantization import (
            quantize_dynamic,
            quantize_static,
            QuantType,
            CalibrationDataReader,
        )
    except ImportError:
        print("  ❌ onnxruntime.quantization not available.")
        print("     Install with: pip install onnxruntime")
        return input_path

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    if calibration_data is not None and len(calibration_data) > 0:
        # Static quantization with calibration data
        class AetherCalibrationReader(CalibrationDataReader):
            def __init__(self, data_list):
                self.data_list = data_list
                self.idx = 0

            def get_next(self):
                if self.idx >= len(self.data_list):
                    return None
                result = {'input': self.data_list[self.idx]}
                self.idx += 1
                return result

        reader = AetherCalibrationReader(calibration_data)
        quantize_static(
            input_path,
            output_path,
            reader,
            quant_format=QuantType.QInt8,
            weight_type=QuantType.QInt8,
        )
        print(f"  ✅ Static INT8 quantization complete: {output_path}")
    else:
        # Dynamic quantization (no calibration needed)
        quantize_dynamic(
            input_path,
            output_path,
            weight_type=QuantType.QInt8,
        )
        print(f"  ✅ Dynamic INT8 quantization complete: {output_path}")

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    orig_size_mb = os.path.getsize(input_path) / (1024 * 1024)
    reduction = (1.0 - file_size_mb / max(orig_size_mb, 0.001)) * 100
    print(f"     Size: {file_size_mb:.2f} MB  ({reduction:.0f}% reduction)")
    return output_path


# ──────────────────────────────────────────────────────────────────────────────
# Full Export Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def export_full_pipeline(checkpoint_path: str,
                         output_dir: str = 'deployment/exported',
                         tile_size: int = 128,
                         latent_dim: int = 128,
                         quantize: bool = True):
    """
    Export both NeuralEngine and GeometricEngine to ONNX with optional INT8.

    Args:
        checkpoint_path : path to trained .pth checkpoint.
        output_dir      : directory for exported models.
        tile_size       : tile edge length (default 128).
        latent_dim      : neural latent dimensionality (default 128).
        quantize        : whether to apply INT8 quantization.
    """
    from src.engines.geometric.surface_fit import GeometricEngine
    from src.engines.neural.lightweight import LightweightNeuralEngine

    print(f"🚀 Aether-Blueprint Mobile Export Pipeline")
    print(f"   Checkpoint: {checkpoint_path}")
    print(f"   Output:     {output_dir}")
    print(f"   Tile Size:  {tile_size}")
    print(f"   Quantize:   {quantize}")
    print()

    device = torch.device('cpu')  # Export always on CPU
    os.makedirs(output_dir, exist_ok=True)

    # ── Build engines ───────────────────────────────────────────────────────
    geo_engine = GeometricEngine(tile_size=tile_size)
    neural_engine = LightweightNeuralEngine(
        tile_size=tile_size, latent_dim=latent_dim
    )

    # ── Load checkpoint (partial, engine weights only) ──────────────────────
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
        # Extract engine weights with flexible key matching
        geo_keys = {k.replace('geo_engine.', ''): v
                    for k, v in state_dict.items() if 'geo_engine' in k or 'geometric_engine' in k}
        neural_keys = {k.replace('neural_engine.', ''): v
                       for k, v in state_dict.items() if 'neural_engine' in k}

        if geo_keys:
            # Clean key names
            clean_geo = {}
            for k, v in geo_keys.items():
                clean_k = k.replace('geometric_engine.', '')
                clean_geo[clean_k] = v
            try:
                geo_engine.load_state_dict(clean_geo, strict=False)
                print(f"  ✅ Loaded GeometricEngine weights ({len(clean_geo)} keys)")
            except Exception as e:
                print(f"  ⚠️ GeometricEngine partial load: {e}")

        if neural_keys:
            try:
                neural_engine.load_state_dict(neural_keys, strict=False)
                print(f"  ✅ Loaded NeuralEngine weights ({len(neural_keys)} keys)")
            except Exception as e:
                print(f"  ⚠️ NeuralEngine partial load: {e}")
    else:
        print(f"  ⚠️ No checkpoint found — exporting with random weights (for testing).")

    print()

    # ── Dummy inputs ────────────────────────────────────────────────────────
    dummy_tile    = torch.randn(1, 3, tile_size, tile_size)
    dummy_latent  = torch.randn(1, latent_dim)
    dummy_coeffs  = torch.randn(1, 3, 10)

    # ── Export Neural Encoder ───────────────────────────────────────────────
    print("── Neural Engine ──")
    neural_enc = NeuralEncoderWrapper(neural_engine)
    neural_enc_path = os.path.join(output_dir, 'neural_encoder.onnx')
    export_to_onnx(
        neural_enc, dummy_tile, neural_enc_path,
        input_names=['tile'], output_names=['latent'],
        dynamic_axes={'tile': {0: 'batch'}, 'latent': {0: 'batch'}},
    )

    # ── Export Neural Decoder ───────────────────────────────────────────────
    neural_dec = NeuralDecoderWrapper(neural_engine)
    neural_dec_path = os.path.join(output_dir, 'neural_decoder.onnx')
    export_to_onnx(
        neural_dec, dummy_latent, neural_dec_path,
        input_names=['latent'], output_names=['tile'],
        dynamic_axes={'latent': {0: 'batch'}, 'tile': {0: 'batch'}},
    )

    # ── Export Geometric Encoder ────────────────────────────────────────────
    print("\n── Geometric Engine ──")
    geo_enc = GeometricEncoderWrapper(geo_engine)
    geo_enc_path = os.path.join(output_dir, 'geometric_encoder.onnx')
    export_to_onnx(
        geo_enc, dummy_tile, geo_enc_path,
        input_names=['tile'], output_names=['coeffs'],
        dynamic_axes={'tile': {0: 'batch'}, 'coeffs': {0: 'batch'}},
    )

    # ── Export Geometric Decoder ────────────────────────────────────────────
    geo_dec = GeometricDecoderWrapper(geo_engine)
    geo_dec_path = os.path.join(output_dir, 'geometric_decoder.onnx')
    export_to_onnx(
        geo_dec, dummy_coeffs, geo_dec_path,
        input_names=['coeffs'], output_names=['tile'],
        dynamic_axes={'coeffs': {0: 'batch'}, 'tile': {0: 'batch'}},
    )

    # ── INT8 Quantization ───────────────────────────────────────────────────
    if quantize:
        print("\n── INT8 Quantization ──")
        for name, fp32_path in [
            ('neural_encoder', neural_enc_path),
            ('neural_decoder', neural_dec_path),
            ('geometric_encoder', geo_enc_path),
            ('geometric_decoder', geo_dec_path),
        ]:
            int8_path = os.path.join(output_dir, f'{name}_int8.onnx')
            quantize_onnx_int8(fp32_path, int8_path)

    # ── Summary ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"🏆 Export Complete!")
    print(f"   Output directory: {output_dir}")
    print(f"   Files exported:")
    for f in sorted(os.listdir(output_dir)):
        if f.endswith('.onnx'):
            size_mb = os.path.getsize(os.path.join(output_dir, f)) / (1024 * 1024)
            print(f"     📦 {f}  ({size_mb:.2f} MB)")
    print(f"{'='*60}")


# ──────────────────────────────────────────────────────────────────────────────
# Convenience: CoreML & TFLite stubs (require additional packages)
# ──────────────────────────────────────────────────────────────────────────────

def export_to_coreml(onnx_path: str, output_path: str):
    """
    Convert ONNX model to CoreML format for iOS deployment.
    Requires: pip install coremltools
    """
    try:
        import coremltools as ct
        import onnx
        onnx_model = onnx.load(onnx_path)
        coreml_model = ct.converters.onnx.convert(onnx_model)
        coreml_model.save(output_path)
        print(f"  ✅ CoreML export: {output_path}")
    except ImportError:
        print("  ❌ coremltools not installed. Run: pip install coremltools")
    except Exception as e:
        print(f"  ❌ CoreML conversion failed: {e}")


def export_to_tflite(onnx_path: str, output_path: str):
    """
    Convert ONNX model to TFLite format for Android deployment.
    Requires: pip install onnx-tf tensorflow
    """
    try:
        import onnx
        from onnx_tf.backend import prepare
        import tensorflow as tf

        onnx_model = onnx.load(onnx_path)
        tf_rep = prepare(onnx_model)
        tf_rep.export_graph(output_path + '_saved_model')

        converter = tf.lite.TFLiteConverter.from_saved_model(output_path + '_saved_model')
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.int8]
        tflite_model = converter.convert()

        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        print(f"  ✅ TFLite export: {output_path}")
    except ImportError:
        print("  ❌ onnx-tf/tensorflow not installed.")
    except Exception as e:
        print(f"  ❌ TFLite conversion failed: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Aether-Blueprint v3.0: Mobile Export Pipeline"
    )
    parser.add_argument('--checkpoint', type=str, default='stage2_perceptual.pth',
                        help='Path to trained .pth checkpoint')
    parser.add_argument('--output_dir', type=str, default='deployment/exported',
                        help='Output directory for exported models')
    parser.add_argument('--tile_size', type=int, default=128,
                        help='Tile size used during training')
    parser.add_argument('--latent_dim', type=int, default=128,
                        help='Neural latent dimensionality')
    parser.add_argument('--no_quantize', action='store_true',
                        help='Skip INT8 quantization')

    args = parser.parse_args()
    export_full_pipeline(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        tile_size=args.tile_size,
        latent_dim=args.latent_dim,
        quantize=not args.no_quantize,
    )
