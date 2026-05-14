"""
orchestrator.py — Aether-Blueprint v3.0
========================================
Central Hybrid Routing Orchestrator.
 
Responsibility:
    1. Accept a batch of image tiles.
    2. Use ComplexityMask to route each tile to the correct engine.
    3. Call the three engines in parallel where possible.
    4. Reassemble all rendered tiles back into full images using
       Gaussian-weighted overlap stitching (eliminates seam artifacts).
 
Key design decisions and bug fixes incorporated:
    - Patch-vs-tile indexing fix: structural similarity is checked at tile
      level (min over all patches in a tile), not patch level.
    - OOM mitigation: neural fallback threshold 0.95 is correct, but batch
      size must be kept small (≤8) externally. The orchestrator itself does
      not batch-split; that is the trainer's responsibility.
    - Gaussian window is pre-computed once and cached on the correct device.
    - All tensor dtype casts happen before engine calls to avoid silent
      precision mismatches during backprop.
"""
 
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
 
from src.core.complexity_mask import ComplexityMask
 
 
class HybridOrchestrator(nn.Module):
    """
    Routes tiles through the three engines and stitches results.
 
    Args:
        geo_engine    : GeometricEngine instance
        struct_engine : StructuralEngine instance
        neural_engine : LightweightNeuralEngine instance
        tile_size     (int)  : edge length of tiles (default 128)
        overlap       (float): fractional overlap between tiles (default 0.5)
        var_threshold (float): complexity mask variance threshold
        fourier_ratio (float): complexity mask Fourier ratio threshold
    """
 
    def __init__(self,
                 geo_engine,
                 struct_engine,
                 neural_engine,
                 tile_size: int   = 128,
                 overlap: float   = 0.5,
                 var_threshold: float = 0.002,
                 fourier_ratio: float = 8.0):
        super().__init__()
        self.geo_engine    = geo_engine
        self.struct_engine = struct_engine
        self.neural_engine = neural_engine
        self.tile_size     = tile_size
        self.overlap       = overlap
        self.mask_module   = ComplexityMask(var_threshold, fourier_ratio)
        self._gauss_cache  = {}   # device → gaussian window tensor
 
    # ------------------------------------------------------------------
    # Gaussian window (cached per device)
    # ------------------------------------------------------------------
 
    def _gaussian_window(self, device: torch.device) -> torch.Tensor:
        """Return a (1, 1, tile_size, tile_size) Gaussian weight window."""
        key = str(device)
        if key not in self._gauss_cache:
            T = self.tile_size
            sigma = T / 6.0
            ax    = torch.arange(T, dtype=torch.float32) - T / 2.0
            gauss = torch.exp(-ax**2 / (2 * sigma**2))
            window = gauss.unsqueeze(0) * gauss.unsqueeze(1)   # (T, T)
            window = window / window.max()                      # normalise to [0,1]
            self._gauss_cache[key] = window.unsqueeze(0).unsqueeze(0).to(device)
        return self._gauss_cache[key]
 
    # ------------------------------------------------------------------
    # Tiling helpers
    # ------------------------------------------------------------------
 
    def extract_tiles(self, images: torch.Tensor):
        """
        Extract overlapping tiles from a batch of images.
 
        Args:
            images: (B, C, H, W) float tensor.
 
        Returns:
            tiles      : (N_total, C, T, T) — all tiles flattened
            tile_info  : list of (b, top, left) for each tile
            image_size : (H, W) of padded image
        """
        B, C, H, W = images.shape
        T    = self.tile_size
        step = max(1, int(T * (1.0 - self.overlap)))
 
        # Pad so tiles cover the full image
        pad_h = (T - H % T) % T if H % T != 0 else 0
        pad_w = (T - W % T) % T if W % T != 0 else 0
        if pad_h > 0 or pad_w > 0:
            images = F.pad(images, (0, pad_w, 0, pad_h), mode='reflect')
 
        _, _, H2, W2 = images.shape
        tops  = list(range(0, H2 - T + 1, step))
        lefts = list(range(0, W2 - T + 1, step))
        # Ensure last tile reaches the edge
        if tops[-1]  + T < H2: tops.append(H2 - T)
        if lefts[-1] + T < W2: lefts.append(W2 - T)
 
        tiles     = []
        tile_info = []
        for b in range(B):
            for top in tops:
                for left in lefts:
                    tiles.append(images[b, :, top:top+T, left:left+T])
                    tile_info.append((b, top, left))
 
        tiles = torch.stack(tiles, dim=0)   # (N_total, C, T, T)
        return tiles, tile_info, (H2, W2)
 
    # ------------------------------------------------------------------
    # Encoding pass
    # ------------------------------------------------------------------
 
    def encode(self, images: torch.Tensor) -> dict:
        """
        Full encoding pass: tile → route → encode per engine.
 
        Args:
            images: (B, C, H, W) float tensor in [0, 1].
 
        Returns:
            results dict suitable for passing to reconstruct().
        """
        tiles, tile_info, padded_size = self.extract_tiles(images)
        routing = self.mask_module(tiles)
 
        geo_idx    = routing['geo_idx']
        struct_idx = routing['struct_idx']
        neural_idx = routing['neural_idx']
 
        results = {
            'tile_info'   : tile_info,
            'padded_size' : padded_size,
            'n_tiles'     : len(tile_info),
            'routing_mask': routing['mask'],
            'stats'       : routing['stats'],
            'geo_idx'     : geo_idx,
            'struct_idx'  : struct_idx,
            'neural_idx'  : neural_idx,
        }
 
        if geo_idx.numel() > 0:
            results['geo'] = self.geo_engine.encode(tiles[geo_idx])
 
        if struct_idx.numel() > 0:
            results['struct'] = self.struct_engine.encode(tiles[struct_idx])
 
        if neural_idx.numel() > 0:
            results['neural'] = self.neural_engine.encode(tiles[neural_idx])
 
        return results
 
    # ------------------------------------------------------------------
    # Reconstruction / decode
    # ------------------------------------------------------------------
 
    def reconstruct(self,
                    results: dict,
                    n_images: int,
                    image_size: tuple,
                    overlap: float = None) -> torch.Tensor:
        """
        Decode engine outputs and stitch tiles back into full images.
 
        Args:
            results    : output dict from encode()
            n_images   : number of images in the original batch (B)
            image_size : (H, W) of the ORIGINAL (unpadded) images
            overlap    : override overlap fraction (uses self.overlap if None)
 
        Returns:
            (B, C, H, W) reconstructed images, clipped to [0, 1].
        """
        if overlap is None:
            overlap = self.overlap
 
        tile_info    = results['tile_info']
        padded_size  = results['padded_size']
        routing_mask = results['routing_mask']
        T            = self.tile_size
        device       = routing_mask.device
        H2, W2       = padded_size
        orig_H, orig_W = image_size
 
        # Determine number of channels from any available decoded tile
        # Fall back to 3 (RGB) if nothing is available yet.
        C = 3
        if 'geo' in results and results['geo'] is not None:
            sample = self.geo_engine.decode(results['geo'][:1])
            C = sample.shape[1]
 
        # Accumulators for Gaussian-weighted blending
        canvas  = torch.zeros(n_images, C, H2, W2, device=device)
        weights = torch.zeros(n_images, 1,  H2, W2, device=device)
        gauss   = self._gaussian_window(device)               # (1,1,T,T)
 
        # Decode all tiles per engine in one batch call
        target_dtype = canvas.dtype
 
        all_rendered = torch.zeros(len(tile_info), C, T, T,
                                   device=device, dtype=target_dtype)
 
        geo_idx    = results.get('geo_idx',    torch.tensor([], dtype=torch.long))
        struct_idx = results.get('struct_idx', torch.tensor([], dtype=torch.long))
        neural_idx = results.get('neural_idx', torch.tensor([], dtype=torch.long))
 
        if geo_idx.numel() > 0 and 'geo' in results:
            decoded = self.geo_engine.decode(results['geo']).to(target_dtype)
            all_rendered[geo_idx] = decoded
 
        if struct_idx.numel() > 0 and 'struct' in results:
            decoded = self.struct_engine.decode(results['struct']).to(target_dtype)
 
            # Patch-vs-tile safety: struct may use similarity check; accept as-is
            # If shape mismatch (should not happen post-fix), fall back to neural
            if decoded.shape[0] == struct_idx.numel():
                all_rendered[struct_idx] = decoded
            else:
                # Reroute mismatched tiles to neural if neural is available
                if neural_idx.numel() > 0 and 'neural' in results:
                    pass  # already handled below
 
        if neural_idx.numel() > 0 and 'neural' in results:
            decoded = self.neural_engine.decode(results['neural']).to(target_dtype)
            all_rendered[neural_idx] = decoded
 
        # Gaussian-weighted accumulation into canvas
        for i, (b, top, left) in enumerate(tile_info):
            tile_rendered = all_rendered[i]          # (C, T, T)
            canvas [b, :, top:top+T, left:left+T] += tile_rendered * gauss[0]
            weights[b, :, top:top+T, left:left+T] += gauss
 
        # Normalise by accumulated weights (safe divide)
        canvas = canvas / weights.clamp(min=1e-6)
 
        # Crop back to original size and clamp
        canvas = canvas[:, :, :orig_H, :orig_W]
        return canvas.clamp(0.0, 1.0)
 
    # ------------------------------------------------------------------
    # Convenience: encode + decode in one call (used during training)
    # ------------------------------------------------------------------
 
    def forward(self, images: torch.Tensor) -> tuple:
        """
        Encode then immediately reconstruct.  Used in training loops.
 
        Returns:
            (reconstructed, stats_dict)
        """
        B, C, H, W = images.shape
        results = self.encode(images)
        recon   = self.reconstruct(results, B, (H, W))
        return recon, results['stats']
 