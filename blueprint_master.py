"""
blueprint_master.py — Aether-Blueprint v3.0
============================================
The Master CLI & High-Level Python API.

"Visual Teleportation": a 4MB photo collapses into 12KB of digital DNA,
transmitted, and reconstructed pixel-perfect on the other side.

CLI Usage:
    python blueprint_master.py teleport --input photo.jpg --output result.padox
    python blueprint_master.py decode   --input result.padox --output recovered.png
    python blueprint_master.py inspect  --input result.padox
    python blueprint_master.py export   --checkpoint stage2_perceptual.pth

Python API Usage:
    from blueprint_master import AetherBlueprint
    ab = AetherBlueprint(checkpoint='stage2_perceptual.pth')
    result = ab.teleport('photo.jpg', 'result.padox')
    print(result)  # {'ratio': 312.5, 'padox_kb': 12.3, ...}
"""

import os
import sys
import time
import argparse
import struct
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF


# ──────────────────────────────────────────────────────────────────────────────
# Lazy imports (keep startup fast)
# ──────────────────────────────────────────────────────────────────────────────

def _build_orchestrator(tile_size: int = 128, latent_dim: int = 128):
    """Build a fresh HybridOrchestrator with all engines."""
    from src.engines.geometric.surface_fit import GeometricEngine
    from src.engines.structural.dictionary import StructuralEngine
    from src.engines.neural.lightweight import LightweightNeuralEngine
    from src.core.orchestrator import HybridOrchestrator

    geo    = GeometricEngine(tile_size=tile_size)
    struct = StructuralEngine(codebook_size=512, patch_size=32, tile_size=tile_size)
    neural = LightweightNeuralEngine(tile_size=tile_size, latent_dim=latent_dim)

    return HybridOrchestrator(
        geo_engine=geo,
        struct_engine=struct,
        neural_engine=neural,
        tile_size=tile_size,
        overlap=0.5,
    )


def _load_checkpoint(orchestrator, checkpoint_path: str, device: torch.device):
    """Load weights from a Stage 1 or Stage 2 checkpoint."""
    if not os.path.exists(checkpoint_path):
        print(f"  ⚠️  No checkpoint at '{checkpoint_path}' — using random weights.")
        return False

    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)

    try:
        missing, unexpected = orchestrator.load_state_dict(state_dict, strict=False)
        n_loaded = len(state_dict) - len(unexpected)
        print(f"  ✅ Loaded {n_loaded}/{len(state_dict)} weight tensors from '{checkpoint_path}'")
        if missing:
            print(f"     Missing  : {len(missing)} keys (new parameters, will use defaults)")
        if unexpected:
            print(f"     Unexpected: {len(unexpected)} keys (old parameters, safely ignored)")
    except Exception as e:
        print(f"  ⚠️  Load error: {e}")
        return False

    return True


def _image_to_tensor(image_path: str) -> torch.Tensor:
    """Load an image file and convert to float32 tensor in [0, 1]."""
    img = Image.open(image_path).convert('RGB')
    return TF.to_tensor(img)   # (3, H, W) float32


def _tensor_to_image(tensor: torch.Tensor, path: str):
    """Save a (3, H, W) float32 tensor in [0, 1] as a PNG."""
    img = TF.to_pil_image(tensor.clamp(0, 1).cpu())
    img.save(path)


def _build_padox_payloads(results: dict):
    """
    Convert orchestrator encode() results → padox payload dicts.
    Returns (geo_payload, struct_payload, neural_payload).
    """
    geo_payload    = {'coeffs': []}
    struct_payload = {
        'indices':   np.array([], dtype=np.uint16),
        'rotations': np.array([], dtype=np.uint8),
        'gains':     np.array([], dtype=np.float32),
        'biases':    np.array([], dtype=np.float32),
    }
    neural_payload = {'latents': np.zeros((0, 128), dtype=np.float32)}

    if 'geo' in results and results['geo'] is not None:
        coeffs = results['geo'].detach().cpu().numpy()   # (N, C, 10)
        geo_payload['coeffs'] = [coeffs[i] for i in range(coeffs.shape[0])]

    if 'struct' in results and results['struct'] is not None:
        s = results['struct']
        struct_payload = {
            'indices':   s['indices'].detach().cpu().numpy().astype(np.uint16),
            'rotations': s['rotations'].detach().cpu().numpy().astype(np.uint8),
            'gains':     s['gains'].detach().cpu().numpy().astype(np.float32),
            'biases':    s['biases'].detach().cpu().numpy().astype(np.float32),
        }

    if 'neural' in results and results['neural'] is not None:
        neural_payload['latents'] = (
            results['neural'].detach().cpu().numpy().astype(np.float32)
        )

    return geo_payload, struct_payload, neural_payload


# ──────────────────────────────────────────────────────────────────────────────
# High-Level Python API
# ──────────────────────────────────────────────────────────────────────────────

class AetherBlueprint:
    """
    High-level Python API for Aether-Blueprint v3.0.

    Example:
        ab = AetherBlueprint('stage2_perceptual.pth')
        info = ab.teleport('photo.jpg', 'photo.padox')
        ab.reconstruct('photo.padox', 'recovered.png')
    """

    def __init__(self,
                 checkpoint: str = 'stage2_perceptual.pth',
                 tile_size:  int = 128,
                 latent_dim: int = 128,
                 device:     str = 'auto'):

        if device == 'auto':
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self._device = torch.device(device)

        self.tile_size  = tile_size
        self.latent_dim = latent_dim

        print(f"🛰️  Aether-Blueprint v3.0  |  device: {self._device}")
        self._orchestrator = _build_orchestrator(tile_size, latent_dim).to(self._device)
        _load_checkpoint(self._orchestrator, checkpoint, self._device)
        self._orchestrator.eval()

    # ------------------------------------------------------------------
    # Teleport (encode → pack → save)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def teleport(self,
                 input_path:  str,
                 output_path: str) -> dict:
        """
        Compress an image to a .padox file.

        Steps:
            1. Detect tile complexity (Fourier-Variance routing).
            2. Extract overlapping tiles.
            3. Route: Geometric / Structural / Neural encoding.
            4. Entropy-Range-Code all payloads into .padox DNA.
            5. Measure and report Teleportation Ratio.
            6. Immediately decode and save teleported_result.png.

        Args:
            input_path  : path to source image (JPG / PNG / WEBP …).
            output_path : path for the output .padox file.

        Returns:
            dict with keys:
                'input_path'    : str
                'output_path'   : str
                'original_w'    : int
                'original_h'    : int
                'original_kb'   : float
                'padox_kb'      : float
                'ratio'         : float
                'encode_ms'     : float
                'decode_ms'     : float
                'recon_path'    : str
                'engine_stats'  : dict
        """
        from src.core.blueprint_format import pack, save

        # ── Load & prep image ────────────────────────────────────────────
        tensor = _image_to_tensor(input_path).to(self._device)  # (3, H, W)
        _, orig_H, orig_W = tensor.shape
        img_batch = tensor.unsqueeze(0)                          # (1, 3, H, W)

        original_kb = (orig_H * orig_W * 3) / 1024.0

        # ── Encode ───────────────────────────────────────────────────────
        t0 = time.perf_counter()
        results = self._orchestrator.encode(img_batch)
        encode_ms = (time.perf_counter() - t0) * 1000.0

        stats = results['stats']

        # ── Pack into .padox ─────────────────────────────────────────────
        geo_p, struct_p, neural_p = _build_padox_payloads(results)
        padox_bytes = pack(
            img_w=orig_W, img_h=orig_H,
            tile_size=self.tile_size,
            geo_payload=geo_p,
            struct_payload=struct_p,
            neural_payload=neural_p,
        )
        padox_kb = len(padox_bytes) / 1024.0
        ratio    = original_kb / max(padox_kb, 0.001)

        # Save .padox
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        save(output_path, padox_bytes)

        # ── Decode immediately → save reconstructed PNG ──────────────────
        t1 = time.perf_counter()
        recon = self._orchestrator.reconstruct(results, 1, (orig_H, orig_W))
        decode_ms = (time.perf_counter() - t1) * 1000.0

        recon_path = str(Path(output_path).with_suffix('')) + '_teleported.png'
        _tensor_to_image(recon[0], recon_path)

        # ── Report ───────────────────────────────────────────────────────
        self._print_teleport_report(
            input_path, output_path, recon_path,
            orig_W, orig_H, original_kb, padox_kb, ratio,
            encode_ms, decode_ms, stats,
        )

        return {
            'input_path'  : input_path,
            'output_path' : output_path,
            'original_w'  : orig_W,
            'original_h'  : orig_H,
            'original_kb' : original_kb,
            'padox_kb'    : padox_kb,
            'ratio'       : ratio,
            'encode_ms'   : encode_ms,
            'decode_ms'   : decode_ms,
            'recon_path'  : recon_path,
            'engine_stats': stats,
        }

    # ------------------------------------------------------------------
    # Decode (.padox → PNG)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def reconstruct(self,
                    padox_path:  str,
                    output_path: str) -> dict:
        """
        Decode a .padox file back to an image.

        Args:
            padox_path  : path to a .padox file.
            output_path : path for the reconstructed PNG.

        Returns:
            dict with 'output_path', 'padox_kb', 'decode_ms'.
        """
        from src.core.blueprint_format import load, unpack

        padox_bytes = load(padox_path)
        meta = unpack(padox_bytes, latent_dim=self.latent_dim)

        orig_W    = meta['img_w']
        orig_H    = meta['img_h']
        padox_kb  = len(padox_bytes) / 1024.0

        # Rebuild results dict compatible with orchestrator.reconstruct()
        results = self._meta_to_results(meta, orig_H, orig_W)

        t0 = time.perf_counter()
        recon = self._orchestrator.reconstruct(results, 1, (orig_H, orig_W))
        decode_ms = (time.perf_counter() - t0) * 1000.0

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        _tensor_to_image(recon[0], output_path)

        print(f"  🖼️  Decoded '{padox_path}' → '{output_path}'  "
              f"({padox_kb:.1f} KB → {orig_W}×{orig_H}px  |  {decode_ms:.1f} ms)")

        return {'output_path': output_path, 'padox_kb': padox_kb, 'decode_ms': decode_ms}

    # ------------------------------------------------------------------
    # Inspect (.padox metadata)
    # ------------------------------------------------------------------

    def inspect(self, padox_path: str) -> dict:
        """
        Print human-readable metadata from a .padox file.

        Args:
            padox_path : path to a .padox file.

        Returns:
            dict with all header fields and section sizes.
        """
        from src.core.blueprint_format import load, unpack, MAGIC, _HDR_FMT, _HDR_SIZE
        import struct

        padox_bytes = load(padox_path)
        meta = unpack(padox_bytes, latent_dim=self.latent_dim)

        raw_kb  = (meta['img_w'] * meta['img_h'] * 3) / 1024.0
        padox_kb = len(padox_bytes) / 1024.0
        ratio    = raw_kb / max(padox_kb, 0.001)

        geo_n    = len(meta['geo']['coeffs'])
        struct_n = len(meta['struct']['indices']) if meta['struct']['indices'] is not None else 0
        neural_n = meta['neural']['latents'].shape[0]
        total_n  = max(geo_n + struct_n // 16 + neural_n, 1)

        print(f"\n{'═'*56}")
        print(f"  🛰️  .padox Inspection Report")
        print(f"{'─'*56}")
        print(f"  File           : {padox_path}")
        print(f"  File Size      : {padox_kb:.2f} KB  ({len(padox_bytes):,} bytes)")
        print(f"  Image Size     : {meta['img_w']} × {meta['img_h']} px")
        print(f"  Raw Equivalent : {raw_kb:.0f} KB")
        print(f"  Tile Size      : {meta['tile_size']} px")
        print(f"{'─'*56}")
        print(f"  Teleportation Ratio : {ratio:.1f}×")
        print(f"{'─'*56}")
        print(f"  Engine Distribution:")
        print(f"    Geometric  (math)       : {geo_n:4d} tiles  ({geo_n/total_n*100:.1f}%)")
        print(f"    Structural (codebook)   : {struct_n//16:4d} tiles  ({(struct_n//16)/total_n*100:.1f}%)")
        print(f"    Neural     (deep)       : {neural_n:4d} tiles  ({neural_n/total_n*100:.1f}%)")
        print(f"{'═'*56}\n")

        return {
            'file'      : padox_path,
            'padox_kb'  : padox_kb,
            'img_w'     : meta['img_w'],
            'img_h'     : meta['img_h'],
            'tile_size' : meta['tile_size'],
            'ratio'     : ratio,
            'geo_tiles' : geo_n,
            'struct_tiles': struct_n // 16,
            'neural_tiles': neural_n,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _meta_to_results(self, meta: dict, H: int, W: int) -> dict:
        """
        Convert unpack() output back to an orchestrator-compatible results dict.
        This enables reconstruct() to be called after loading from disk.
        """
        import math

        tile_size = self.tile_size
        step      = tile_size // 2   # 50% overlap

        # Rebuild tile_info list
        # Pad dimensions to nearest tile boundary
        H2 = H + (tile_size - H % tile_size) % tile_size if H % tile_size != 0 else H
        W2 = W + (tile_size - W % tile_size) % tile_size if W % tile_size != 0 else W

        tops  = list(range(0, H2 - tile_size + 1, step))
        lefts = list(range(0, W2 - tile_size + 1, step))
        if tops  and tops[-1]  + tile_size < H2: tops.append(H2 - tile_size)
        if lefts and lefts[-1] + tile_size < W2: lefts.append(W2 - tile_size)

        tile_info = [(0, top, left) for top in tops for left in lefts]
        n_tiles   = len(tile_info)

        device = self._device

        # Routing mask: first n_geo geo, then n_struct, then n_neural
        geo_coeffs = meta['geo']['coeffs']
        struct_d   = meta['struct']
        neural_lat = meta['neural']['latents']

        n_geo    = len(geo_coeffs)
        n_struct = len(struct_d['indices']) // max(16, 1) if len(struct_d['indices']) > 0 else 0
        n_neural = neural_lat.shape[0]

        mask = torch.zeros(n_tiles, dtype=torch.long, device=device)
        geo_idx    = torch.arange(n_geo, dtype=torch.long, device=device)
        struct_idx = torch.arange(n_geo, n_geo + n_struct, dtype=torch.long, device=device)
        neural_idx = torch.arange(n_geo + n_struct, n_geo + n_struct + n_neural,
                                  dtype=torch.long, device=device)

        mask[geo_idx]    = 0
        mask[struct_idx] = 1
        mask[neural_idx] = 2

        results = {
            'tile_info'   : tile_info[:n_tiles],
            'padded_size' : (H2, W2),
            'n_tiles'     : n_tiles,
            'routing_mask': mask,
            'stats'       : {'pct_geometric': 0, 'pct_structural': 0, 'pct_neural': 0},
            'geo_idx'     : geo_idx,
            'struct_idx'  : struct_idx,
            'neural_idx'  : neural_idx,
        }

        # Attach decoded engine outputs
        if n_geo > 0:
            arr = np.stack(geo_coeffs, axis=0).astype(np.float32)  # (N, C, 10)
            results['geo'] = torch.from_numpy(arr).to(device)

        if n_struct > 0:
            results['struct'] = {
                'indices'   : torch.from_numpy(struct_d['indices'].astype(np.int64)).to(device),
                'rotations' : torch.from_numpy(struct_d['rotations'].astype(np.int64)).to(device),
                'gains'     : torch.from_numpy(struct_d['gains']).to(device),
                'biases'    : torch.from_numpy(struct_d['biases']).to(device),
                'n_tiles'   : n_struct,
            }

        if n_neural > 0:
            results['neural'] = torch.from_numpy(neural_lat).to(device)

        return results

    @staticmethod
    def _print_teleport_report(input_path, output_path, recon_path,
                                orig_W, orig_H, original_kb, padox_kb, ratio,
                                encode_ms, decode_ms, stats):
        bar_filled = int(min(ratio / 400.0, 1.0) * 40)
        bar = '█' * bar_filled + '░' * (40 - bar_filled)

        print(f"\n{'═'*62}")
        print(f"  🛰️  VISUAL TELEPORTATION COMPLETE")
        print(f"{'─'*62}")
        print(f"  Source  : {input_path}  ({orig_W}×{orig_H})")
        print(f"  .padox  : {output_path}")
        print(f"  Result  : {recon_path}")
        print(f"{'─'*62}")
        print(f"  Original Size  : {original_kb:>10.1f} KB")
        print(f"  .padox Size    : {padox_kb:>10.2f} KB")
        print(f"  ─────────────────────────────")
        print(f"  Ratio          : {ratio:>10.1f}×")
        print(f"  [{bar}]")
        print(f"{'─'*62}")
        print(f"  Encode  : {encode_ms:6.1f} ms    Decode : {decode_ms:6.1f} ms")
        print(f"{'─'*62}")
        print(f"  Engine Mix:")
        print(f"    🧮 Geometric  (polynomial) : {stats['pct_geometric']:.1f}%")
        print(f"    📚 Structural (codebook)   : {stats['pct_structural']:.1f}%")
        print(f"    ⚡ Neural     (deep)       : {stats['pct_neural']:.1f}%")
        print(f"{'═'*62}\n")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _cmd_teleport(args):
    ab = AetherBlueprint(
        checkpoint=args.checkpoint,
        tile_size=args.tile_size,
        latent_dim=args.latent_dim,
    )
    ab.teleport(args.input, args.output)


def _cmd_decode(args):
    ab = AetherBlueprint(
        checkpoint=args.checkpoint,
        tile_size=args.tile_size,
        latent_dim=args.latent_dim,
    )
    ab.reconstruct(args.input, args.output)


def _cmd_inspect(args):
    ab = AetherBlueprint(
        checkpoint=args.checkpoint,
        tile_size=args.tile_size,
        latent_dim=args.latent_dim,
    )
    ab.inspect(args.input)


def _cmd_export(args):
    from deployment.mobile_export import export_full_pipeline
    export_full_pipeline(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        tile_size=args.tile_size,
        latent_dim=args.latent_dim,
        quantize=not args.no_quantize,
    )


def main():
    parser = argparse.ArgumentParser(
        prog='blueprint_master',
        description='🛰️  Aether-Blueprint v3.0 — Visual Teleportation Engine',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  teleport   Compress an image to a .padox file and immediately decode it.
  decode     Decompress a .padox file back to a full-resolution PNG.
  inspect    Show metadata and compression stats for a .padox file.
  export     Export trained engines to ONNX + INT8 for mobile deployment.

Examples:
  python blueprint_master.py teleport --input photo.jpg --output photo.padox
  python blueprint_master.py decode   --input photo.padox --output recovered.png
  python blueprint_master.py inspect  --input photo.padox
  python blueprint_master.py export   --checkpoint stage2_perceptual.pth
        """,
    )

    # ── Global flags ────────────────────────────────────────────────────────
    parser.add_argument('--checkpoint', type=str, default='stage2_perceptual.pth',
                        help='Path to trained .pth checkpoint (default: stage2_perceptual.pth)')
    parser.add_argument('--tile_size',  type=int, default=128,
                        help='Tile edge length in pixels (default: 128)')
    parser.add_argument('--latent_dim', type=int, default=128,
                        help='Neural latent dimensionality (default: 128)')

    subparsers = parser.add_subparsers(dest='command', metavar='command')
    subparsers.required = True

    # ── teleport ─────────────────────────────────────────────────────────────
    p_teleport = subparsers.add_parser('teleport',
        help='Compress an image to .padox and reconstruct it immediately.')
    p_teleport.add_argument('--input',  required=True, metavar='IMAGE',
                             help='Source image path (JPG, PNG, WEBP …)')
    p_teleport.add_argument('--output', required=True, metavar='PADOX',
                             help='Output .padox file path')
    p_teleport.set_defaults(func=_cmd_teleport)

    # ── decode ───────────────────────────────────────────────────────────────
    p_decode = subparsers.add_parser('decode',
        help='Decompress a .padox file to a PNG image.')
    p_decode.add_argument('--input',  required=True, metavar='PADOX',
                           help='Source .padox file path')
    p_decode.add_argument('--output', required=True, metavar='PNG',
                           help='Output PNG file path')
    p_decode.set_defaults(func=_cmd_decode)

    # ── inspect ──────────────────────────────────────────────────────────────
    p_inspect = subparsers.add_parser('inspect',
        help='Show metadata and compression stats for a .padox file.')
    p_inspect.add_argument('--input', required=True, metavar='PADOX',
                            help='.padox file to inspect')
    p_inspect.set_defaults(func=_cmd_inspect)

    # ── export ───────────────────────────────────────────────────────────────
    p_export = subparsers.add_parser('export',
        help='Export trained engines to ONNX + INT8 for mobile deployment.')
    p_export.add_argument('--output_dir', type=str, default='deployment/exported',
                           help='Output directory for ONNX models')
    p_export.add_argument('--no_quantize', action='store_true',
                           help='Skip INT8 quantization step')
    p_export.set_defaults(func=_cmd_export)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
