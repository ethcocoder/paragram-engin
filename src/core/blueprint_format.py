"""
Aether-Blueprint v3.0: Blueprint Format (The Binary Vault)
---------------------------------------------------------
Custom .padox binary specification. Implements low-level bit-packing
for maximum compression efficiency.

Binary Layout:
  ┌────────────────────────────────────────────────┐
  │  HEADER (16 bytes)                             │
  │  ─────────────────                             │
  │  Magic     : 5 bytes  "PADOX"                  │
  │  Version   : 1 byte   (0x03 for v3.0)          │
  │  Width     : 2 bytes  uint16                    │
  │  Height    : 2 bytes  uint16                    │
  │  TileSize  : 1 byte   uint8                     │
  │  NumTiles  : 2 bytes  uint16                    │
  │  Overlap   : 1 byte   uint8                     │
  │  Reserved  : 2 bytes  (future use)              │
  ├────────────────────────────────────────────────┤
  │  MASK SECTION                                  │
  │  2 bits per tile, bit-packed into bytes         │
  ├────────────────────────────────────────────────┤
  │  GEOMETRIC PAYLOAD (State 0 tiles)             │
  │  int8 quantized coefficients (3 ch × 10 coeff) │
  ├────────────────────────────────────────────────┤
  │  STRUCTURAL PAYLOAD (State 1 tiles)            │
  │  int16 indices + int8 rotations/gains/biases   │
  ├────────────────────────────────────────────────┤
  │  NEURAL PAYLOAD (State 2 tiles)                │
  │  Zlib-compressed FP16 latents                  │
  └────────────────────────────────────────────────┘
"""
import struct
import zlib
import io
import numpy as np
import torch


# ─── Constants ───────────────────────────────────────────────────────────────
PADOX_MAGIC = b"PADOX"
PADOX_VERSION = 3
HEADER_SIZE = 16  # bytes


# ─── Packer ──────────────────────────────────────────────────────────────────

def pack_blueprint(results, image_width, image_height, tile_size=128, overlap=64):
    """
    Packs orchestrator results into a compact .padox binary byte-array.

    Args:
        results: dict from AetherOrchestrator.forward() containing
                 'mask', 'geometric', 'structural', 'neural'.
        image_width:  original image width in pixels.
        image_height: original image height in pixels.
        tile_size:    tile edge length (default 128).
        overlap:      overlap in pixels (default 64).

    Returns:
        bytes: the packed .padox payload.
    """
    buf = io.BytesIO()

    mask_tensor = results['mask'].view(-1).cpu()   # (num_tiles,)
    num_tiles = mask_tensor.numel()

    # ── 1. Header (16 bytes) ────────────────────────────────────────────────
    buf.write(PADOX_MAGIC)                                       # 5 B
    buf.write(struct.pack('B', PADOX_VERSION))                   # 1 B
    buf.write(struct.pack('>H', image_width))                    # 2 B
    buf.write(struct.pack('>H', image_height))                   # 2 B
    buf.write(struct.pack('B', tile_size))                       # 1 B
    buf.write(struct.pack('>H', num_tiles))                      # 2 B
    buf.write(struct.pack('B', overlap))                         # 1 B
    buf.write(b'\x00\x00')                                       # 2 B reserved

    # ── 2. Mask section (2 bits per tile, packed into bytes) ────────────────
    mask_np = mask_tensor.numpy().astype(np.uint8)
    packed_mask = _bitpack_mask(mask_np)
    buf.write(struct.pack('>H', len(packed_mask)))
    buf.write(packed_mask)

    # ── 3. Geometric payload ────────────────────────────────────────────────
    if results['geometric'] is not None:
        coeffs = results['geometric'].detach().cpu()          # (B_g, 3, 10)
        q_coeffs = torch.round(coeffs * 127.0).clamp(-128, 127).to(torch.int8)
        raw = q_coeffs.numpy().tobytes()
        buf.write(struct.pack('>I', len(raw)))
        buf.write(raw)
    else:
        buf.write(struct.pack('>I', 0))

    # ── 4. Structural payload ───────────────────────────────────────────────
    if results['structural'] is not None:
        indices, rots, gains, biases, _sims = results['structural']
        indices_np = indices.detach().cpu().to(torch.int16).numpy()
        rots_np    = rots.detach().cpu().to(torch.int8).numpy()
        gains_np   = torch.round(gains.detach().cpu() * 127.0).clamp(-128, 127).to(torch.int8).numpy()
        biases_np  = torch.round(biases.detach().cpu() * 127.0).clamp(-128, 127).to(torch.int8).numpy()

        # Concatenate into a single buffer: indices(2B each) + rots(1B) + gains(1B) + biases(1B)
        s_buf = io.BytesIO()
        s_buf.write(struct.pack('>I', len(indices_np)))  # num patches
        s_buf.write(indices_np.tobytes())
        s_buf.write(rots_np.tobytes())
        s_buf.write(gains_np.tobytes())
        s_buf.write(biases_np.tobytes())
        s_raw = s_buf.getvalue()
        buf.write(struct.pack('>I', len(s_raw)))
        buf.write(s_raw)
    else:
        buf.write(struct.pack('>I', 0))

    # ── 5. Neural payload (zlib-compressed FP16 latents) ────────────────────
    if results['neural'] is not None:
        latent = results['neural'].detach().cpu().to(torch.float16)
        raw_latent = latent.numpy().tobytes()
        compressed = zlib.compress(raw_latent, level=6)
        # Store shape header so we can reconstruct later
        shape_bytes = struct.pack('>4I', *latent.shape) if latent.dim() == 4 else struct.pack('>I', latent.numel())
        buf.write(struct.pack('>I', len(shape_bytes) + len(compressed)))
        buf.write(shape_bytes)
        buf.write(compressed)
    else:
        buf.write(struct.pack('>I', 0))

    return buf.getvalue()


# ─── Unpacker ────────────────────────────────────────────────────────────────

def unpack_blueprint(data):
    """
    Unpacks a .padox binary byte-array back into orchestrator-compatible results.

    Args:
        data: bytes from pack_blueprint().

    Returns:
        dict with keys:
            'mask'       : LongTensor (num_tiles,)
            'geometric'  : FloatTensor (B_g, 3, 10) or None
            'structural' : tuple(indices, rots, gains, biases, None) or None
            'neural'     : FloatTensor (B_n, C, H, W) or None
            'meta'       : dict with width, height, tile_size, overlap, num_tiles
    """
    buf = io.BytesIO(data)

    # ── 1. Header ───────────────────────────────────────────────────────────
    magic = buf.read(5)
    assert magic == PADOX_MAGIC, f"Invalid magic: {magic}"
    version   = struct.unpack('B', buf.read(1))[0]
    width     = struct.unpack('>H', buf.read(2))[0]
    height    = struct.unpack('>H', buf.read(2))[0]
    tile_size = struct.unpack('B', buf.read(1))[0]
    num_tiles = struct.unpack('>H', buf.read(2))[0]
    overlap   = struct.unpack('B', buf.read(1))[0]
    _reserved = buf.read(2)

    meta = dict(version=version, width=width, height=height,
                tile_size=tile_size, num_tiles=num_tiles, overlap=overlap)

    # ── 2. Mask ─────────────────────────────────────────────────────────────
    mask_len = struct.unpack('>H', buf.read(2))[0]
    packed_mask = buf.read(mask_len)
    mask_np = _bitunpack_mask(packed_mask, num_tiles)
    mask = torch.from_numpy(mask_np).long()

    results = {'mask': mask, 'geometric': None, 'structural': None, 'neural': None, 'meta': meta}

    # ── 3. Geometric ────────────────────────────────────────────────────────
    geo_len = struct.unpack('>I', buf.read(4))[0]
    if geo_len > 0:
        raw = buf.read(geo_len)
        q_coeffs = np.frombuffer(raw, dtype=np.int8).copy()
        num_geo = geo_len // (3 * 10)
        q_coeffs = q_coeffs.reshape(num_geo, 3, 10)
        results['geometric'] = torch.from_numpy(q_coeffs).float() / 127.0

    # ── 4. Structural ───────────────────────────────────────────────────────
    struct_len = struct.unpack('>I', buf.read(4))[0]
    if struct_len > 0:
        s_buf = io.BytesIO(buf.read(struct_len))
        num_patches = struct.unpack('>I', s_buf.read(4))[0]
        indices = torch.from_numpy(np.frombuffer(s_buf.read(num_patches * 2), dtype=np.int16).copy()).long()
        rots    = torch.from_numpy(np.frombuffer(s_buf.read(num_patches), dtype=np.int8).copy()).long()
        gains   = torch.from_numpy(np.frombuffer(s_buf.read(num_patches), dtype=np.int8).copy()).float() / 127.0
        biases  = torch.from_numpy(np.frombuffer(s_buf.read(num_patches), dtype=np.int8).copy()).float() / 127.0
        results['structural'] = (indices, rots, gains, biases, None)

    # ── 5. Neural ───────────────────────────────────────────────────────────
    neural_len = struct.unpack('>I', buf.read(4))[0]
    if neural_len > 0:
        shape_bytes = buf.read(16)  # 4 × uint32
        shape = struct.unpack('>4I', shape_bytes)
        compressed = buf.read(neural_len - 16)
        raw_latent = zlib.decompress(compressed)
        latent_np = np.frombuffer(raw_latent, dtype=np.float16).copy().reshape(shape)
        results['neural'] = torch.from_numpy(latent_np).float()

    return results


# ─── Bit-pack helpers ────────────────────────────────────────────────────────

def _bitpack_mask(mask_np):
    """Pack an array of 2-bit values (0-2) into bytes, 4 values per byte."""
    n = len(mask_np)
    # Pad to multiple of 4
    padded = np.zeros(((n + 3) // 4) * 4, dtype=np.uint8)
    padded[:n] = mask_np
    packed = np.zeros(len(padded) // 4, dtype=np.uint8)
    for i in range(4):
        packed |= (padded[i::4] & 0x03) << (6 - 2 * i)
    return packed.tobytes()


def _bitunpack_mask(raw_bytes, num_tiles):
    """Unpack bytes into an array of 2-bit values."""
    packed = np.frombuffer(raw_bytes, dtype=np.uint8)
    result = np.zeros(len(packed) * 4, dtype=np.uint8)
    for i in range(4):
        result[i::4] = (packed >> (6 - 2 * i)) & 0x03
    return result[:num_tiles]


# ─── File I/O convenience ───────────────────────────────────────────────────

def save_padox(filepath, data_bytes):
    """Write packed bytes to a .padox file."""
    with open(filepath, 'wb') as f:
        f.write(data_bytes)


def load_padox(filepath):
    """Read a .padox file and return unpacked results."""
    with open(filepath, 'rb') as f:
        return unpack_blueprint(f.read())
