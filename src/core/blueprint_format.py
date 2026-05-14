"""
blueprint_format.py — Aether-Blueprint v3.0
============================================
Custom bit-packed .padox binary specification.
 
Responsibility:
    Pack and unpack the full hybrid payload for a single compressed image into
    the .padox container format.  The format stores three payload sections
    (one per engine) plus a fixed header, all in a compact binary stream.
 
    v2 upgrade: Neural and Geometric payloads now use the Vectorized Range
    Coder (EntropyCoder) for 30–50% better compression than zlib alone.

.padox Layout v2
─────────────────
    [MAGIC]      4 bytes  — b'PADX'
    [VERSION]    1 byte   — format version (2 = entropy-coded)
    [IMG_W]      2 bytes  — original image width  (uint16)
    [IMG_H]      2 bytes  — original image height (uint16)
    [TILE_SIZE]  2 bytes  — tile size used (uint16)
    [N_GEO]      4 bytes  — number of geometric tiles (uint32)
    [N_STR]      4 bytes  — number of structural patches (uint32)
    [N_NEU]      4 bytes  — number of neural tiles (uint32)
    [GEO_BYTES]  4 bytes  — byte length of geometric payload (uint32)
    [STR_BYTES]  4 bytes  — byte length of structural payload (uint32)
    [NEU_BYTES]  4 bytes  — byte length of neural payload (uint32)
    ── payload sections ──────────────────────────────────────────────
    [GEO_DATA]   raw bytes — entropy-coded polynomial coefficients
    [STR_DATA]   raw bytes — tight-packed codebook records (already compact)
    [NEU_DATA]   raw bytes — entropy-coded latent vectors
 
Design notes:
    - All multi-byte integers are little-endian.
    - The geometric payload uses entropy coding (replaces raw float32).
    - The structural payload uses tight packing: index (uint16) + rotation (uint8)
      + gain (float16) + bias (float16) = 7 bytes per patch.
    - The neural payload uses entropy coding (replaces zlib-compressed FP16).
    - Tile index arrays (which tiles went to which engine) are stored implicitly
      by the N_* counts; the caller must track ordering.
"""
 
import io
import struct
import zlib
import numpy as np
import torch

from src.utils.entropy_coder import EntropyCoder, entropy_encode_tensor
 
MAGIC   = b'PADX'
VERSION = 2          # v2: entropy-coded payloads
 
# Header struct: magic(4) ver(1) w(2) h(2) tile(2) n_geo(4) n_str(4) n_neu(4)
#                geo_bytes(4) str_bytes(4) neu_bytes(4)  → 35 bytes total
_HDR_FMT  = '<4sBHHHIIIIII'
_HDR_SIZE = struct.calcsize(_HDR_FMT)
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Geometric payload helpers
# ──────────────────────────────────────────────────────────────────────────────
 
def _pack_geo(coeffs_list: list) -> bytes:
    """
    Pack geometric coefficients using entropy coding.

    Args:
        coeffs_list: list of numpy arrays, each (C, 10) float32.
    Returns:
        Entropy-coded bytes (30–50% smaller than raw float32).
    """
    if not coeffs_list:
        return b''
    arr = np.stack(coeffs_list, axis=0).astype(np.float32)  # (N, C, 10)
    coder = EntropyCoder(n_bins=256)
    return coder.encode(arr)
 
 
def _unpack_geo(data: bytes, n: int, channels: int = 3) -> list:
    """
    Unpack entropy-coded geometric payload back into coefficient arrays.
    """
    if n == 0 or len(data) == 0:
        return []
    coder = EntropyCoder()
    arr = coder.decode(data)
    arr = arr.reshape(n, channels, 10)
    return [arr[i] for i in range(n)]
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Structural payload helpers
# ──────────────────────────────────────────────────────────────────────────────
# Per-patch record: index(uint16=2) + rotation(uint8=1) + gain(fp16=2) + bias(fp16=2)
_PATCH_RECORD_BYTES = 7
 
def _pack_struct(indices: np.ndarray,
                 rotations: np.ndarray,
                 gains: np.ndarray,
                 biases: np.ndarray) -> bytes:
    """
    Tightly pack structural engine outputs (vectorized, no per-element loops).
 
    Args:
        indices   : (N,) uint16 — codebook index per patch
        rotations : (N,) uint8  — rotation id (0-3)
        gains     : (N,) float32 — affine gain, stored as float16
        biases    : (N,) float32 — affine bias, stored as float16
    Returns:
        Raw bytes.
    """
    if len(indices) == 0:
        return b''
    # Vectorized type conversion
    idx_u16  = indices.astype(np.uint16)
    rot_u8   = rotations.astype(np.uint8)
    gain_f16 = gains.astype(np.float16)
    bias_f16 = biases.astype(np.float16)
    # Concatenate all arrays sequentially (no per-element loop)
    buf = io.BytesIO()
    buf.write(idx_u16.tobytes())
    buf.write(rot_u8.tobytes())
    buf.write(gain_f16.tobytes())
    buf.write(bias_f16.tobytes())
    return buf.getvalue()
 
 
def _unpack_struct(data: bytes, n_patches: int) -> dict:
    """
    Unpack structural payload (vectorized).
    """
    if n_patches == 0 or len(data) == 0:
        return {'indices': np.array([], dtype=np.uint16),
                'rotations': np.array([], dtype=np.uint8),
                'gains': np.array([], dtype=np.float32),
                'biases': np.array([], dtype=np.float32)}
    
    # Calculate offsets for each array
    idx_bytes = n_patches * 2   # uint16
    rot_bytes = n_patches * 1   # uint8
    g_bytes   = n_patches * 2   # float16
    b_bytes   = n_patches * 2   # float16
    
    offset = 0
    indices   = np.frombuffer(data[offset:offset + idx_bytes], dtype=np.uint16).copy()
    offset += idx_bytes
    rotations = np.frombuffer(data[offset:offset + rot_bytes], dtype=np.uint8).copy()
    offset += rot_bytes
    gains     = np.frombuffer(data[offset:offset + g_bytes], dtype=np.float16).copy().astype(np.float32)
    offset += g_bytes
    biases    = np.frombuffer(data[offset:offset + b_bytes], dtype=np.float16).copy().astype(np.float32)
    
    return {'indices': indices, 'rotations': rotations,
            'gains': gains, 'biases': biases}
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Neural payload helpers
# ──────────────────────────────────────────────────────────────────────────────
 
def _pack_neural(latents: np.ndarray) -> bytes:
    """
    Compress neural latents using the Vectorized Range Coder.

    This replaces the previous zlib pipeline, achieving 30–50% better
    compression on structured latent distributions.
 
    Args:
        latents: (N, D) float32 numpy array.
    Returns:
        Entropy-coded bytes.
    """
    if latents.size == 0:
        return b''
    coder = EntropyCoder(n_bins=256)
    return coder.encode(latents.astype(np.float32))
 
 
def _unpack_neural(data: bytes, n: int, latent_dim: int) -> np.ndarray:
    """
    Decompress entropy-coded neural latents.
    Returns (N, latent_dim) float32 array.
    """
    if n == 0 or len(data) == 0:
        return np.zeros((0, latent_dim), dtype=np.float32)
    coder = EntropyCoder()
    arr = coder.decode(data)
    return arr.reshape(n, latent_dim).astype(np.float32)
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────
 
def pack(img_w: int,
         img_h: int,
         tile_size: int,
         geo_payload: dict,
         struct_payload: dict,
         neural_payload: dict) -> bytes:
    """
    Serialize a complete hybrid payload to .padox bytes.
 
    Args:
        img_w, img_h  : original image dimensions
        tile_size     : tile edge length used during encoding
        geo_payload   : {'coeffs': list of (C,10) float32 arrays}
        struct_payload: {'indices': (N,) uint16, 'rotations': (N,) uint8,
                         'gains': (N,) float32, 'biases': (N,) float32}
                        N = n_struct_tiles * patches_per_tile
        neural_payload: {'latents': (N, D) float32 array}
 
    Returns:
        Raw .padox bytes.
    """
    geo_bytes    = _pack_geo(geo_payload.get('coeffs', []))
    struct_bytes = _pack_struct(
        struct_payload.get('indices',   np.array([], dtype=np.uint16)),
        struct_payload.get('rotations', np.array([], dtype=np.uint8)),
        struct_payload.get('gains',     np.array([], dtype=np.float32)),
        struct_payload.get('biases',    np.array([], dtype=np.float32)),
    )
    neural_bytes = _pack_neural(neural_payload.get('latents',
                                 np.zeros((0, 1), dtype=np.float32)))
 
    n_geo    = len(geo_payload.get('coeffs', []))
    n_str    = len(struct_payload.get('indices', [])) if struct_payload.get('indices') is not None else 0
    n_neu    = neural_payload.get('latents', np.zeros((0,1))).shape[0]
 
    header = struct.pack(
        _HDR_FMT,
        MAGIC, VERSION,
        img_w, img_h, tile_size,
        n_geo, n_str, n_neu,
        len(geo_bytes), len(struct_bytes), len(neural_bytes),
    )
    return header + geo_bytes + struct_bytes + neural_bytes
 
 
def unpack(data: bytes, channels: int = 3, latent_dim: int = 128) -> dict:
    """
    Deserialize .padox bytes into payload dictionaries.
 
    Args:
        data       : raw .padox bytes
        channels   : number of image channels (default 3 for RGB)
        latent_dim : neural latent vector dimensionality
 
    Returns:
        dict with keys:
            'img_w', 'img_h', 'tile_size'
            'geo'    : {'coeffs': list}
            'struct' : {'indices', 'rotations', 'gains', 'biases'}
            'neural' : {'latents': ndarray}
    """
    if len(data) < _HDR_SIZE:
        raise ValueError(f"Data too short to be a valid .padox file ({len(data)} bytes).")
 
    (magic, version, img_w, img_h, tile_size,
     n_geo, n_str, n_neu,
     geo_len, str_len, neu_len) = struct.unpack_from(_HDR_FMT, data, 0)
 
    if magic != MAGIC:
        raise ValueError(f"Invalid magic bytes: {magic!r} (expected {MAGIC!r})")
    if version not in (1, 2):
        raise ValueError(f"Unsupported .padox version: {version}")
 
    cursor  = _HDR_SIZE
    geo_raw = data[cursor : cursor + geo_len];  cursor += geo_len
    str_raw = data[cursor : cursor + str_len];  cursor += str_len
    neu_raw = data[cursor : cursor + neu_len]
 
    return {
        'img_w'    : img_w,
        'img_h'    : img_h,
        'tile_size': tile_size,
        'geo'      : {'coeffs': _unpack_geo(geo_raw, n_geo, channels)},
        'struct'   : _unpack_struct(str_raw, n_str),
        'neural'   : {'latents': _unpack_neural(neu_raw, n_neu, latent_dim)},
    }
 
 
def save(path: str, data: bytes) -> None:
    """Write .padox bytes to disk."""
    with open(path, 'wb') as f:
        f.write(data)
 
 
def load(path: str) -> bytes:
    """Read .padox bytes from disk."""
    with open(path, 'rb') as f:
        return f.read()
 
 
def compression_ratio(original_w: int, original_h: int,
                      padox_bytes: bytes, channels: int = 3) -> float:
    """Return the compression ratio (original / compressed)."""
    original_size = original_w * original_h * channels  # bytes at uint8
    return original_size / max(len(padox_bytes), 1)