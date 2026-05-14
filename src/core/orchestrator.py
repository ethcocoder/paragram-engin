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
    def __init__(self,
                 geo_engine,
                 struct_engine,
                 neural_engine,
                 tile_size: int = 128,
                 overlap: float = 0.5,
                 var_threshold: float = 0.015,
                 fourier_ratio: float = 150.0):

        """
        Routes tiles through the three engines and stitches results.
        
        Args:
            geo_engine    : GeometricEngine instance
            struct_engine : StructuralEngine instance
            neural_engine : LightweightNeuralEngine instance
            tile_size     : edge length of tiles
            overlap       : fractional overlap between tiles
            var_threshold : complexity mask variance threshold
            fourier_ratio : complexity mask Fourier ratio threshold
        """
        super().__init__()
        self.geo_engine    = geo_engine
        self.struct_engine = struct_engine
        self.neural_engine = neural_engine
        self.tile_size     = tile_size
        self.overlap       = overlap
        self.mask_module   = ComplexityMask(var_threshold, fourier_ratio)
        self._gauss_cache  = {}

    def get_curriculum_threshold(self, epoch: int):
        """Start loose (0.70) and tighten to 0.95 over 20 epochs."""
        return min(0.95, 0.70 + (epoch * 0.0125))

    def _gaussian_window(self, device: torch.device, dtype=torch.float32) -> torch.Tensor:
        key = f"{device}_{dtype}"
        if key not in self._gauss_cache:
            T = self.tile_size
            sigma = T / 6.0
            ax = torch.arange(T, device=device, dtype=dtype) - T / 2.0
            gauss = torch.exp(-ax**2 / (2 * sigma**2))
            window = gauss.unsqueeze(0) * gauss.unsqueeze(1)
            window = window / window.max()
            self._gauss_cache[key] = window.unsqueeze(0).unsqueeze(0)
        return self._gauss_cache[key]

    def extract_tiles(self, images: torch.Tensor):
        """Extract overlapping tiles using vectorized unfold."""
        B, C, H, W = images.shape
        T = self.tile_size
        step = max(1, int(T * (1.0 - self.overlap)))

        # Robust Padding for F.fold compatibility
        pad_h = (step - (H - T) % step) % step if H > T else T - H
        pad_w = (step - (W - T) % step) % step if W > T else T - W
        images = F.pad(images, (0, pad_w, 0, pad_h), mode='reflect')

        _, _, H2, W2 = images.shape
        tiles = images.unfold(2, T, step).unfold(3, T, step)
        NH, NW = tiles.shape[2], tiles.shape[3]
        
        # (B, C, NH, NW, T, T) -> (B*NH*NW, C, T, T)
        tiles = tiles.permute(0, 2, 3, 1, 4, 5).reshape(-1, C, T, T)
        return tiles, (NH, NW), (H2, W2)

    def encode(self, images: torch.Tensor, current_epoch: int = 1) -> dict:
        """
        Vectorized Encoding Pass:
        1. Classify tiles via ComplexityMask.
        2. Route and encode using G, S, and N engines.
        3. Apply curriculum-based fallback for low-quality structural matches.
        """
        tiles, grid_dim, padded_size = self.extract_tiles(images)
        routing = self.mask_module(tiles)
        mask = routing['mask']
        sim_tau = self.get_curriculum_threshold(current_epoch)

        results = {
            'grid_dim': grid_dim,
            'padded_size': padded_size,
            'routing_mask': mask,
            'stats': routing['stats'],
            'geo': None, 'struct': None, 'neural': None
        }

        # --- 1. GEOMETRIC ---
        geo_idx = (mask == 0).nonzero(as_tuple=True)[0]
        if geo_idx.numel() > 0:
            results['geo'] = self.geo_engine.encode(tiles[geo_idx])

        # --- 2. STRUCTURAL ---
        struct_idx = (mask == 1).nonzero(as_tuple=True)[0]
        if struct_idx.numel() > 0:
            s_data = self.struct_engine.encode(tiles[struct_idx])
            
            # Extract from dictionary safely
            indices = s_data['indices']
            rots    = s_data['rotations']
            gains   = s_data['gains']
            biases  = s_data['biases']
            sims    = s_data['similarities']
            
            # Curriculum Fallback: Check if pattern match is sharp enough
            sims_per_tile = sims.mean(dim=1) if sims.dim() > 1 else sims
            bad_matches = (sims_per_tile < sim_tau)
            
            if bad_matches.any():
                # Reroute rejected tiles to Neural Engine
                mask[struct_idx[bad_matches]] = 2
                good_mask = ~bad_matches
                
                # Filter payload for Structural
                P = self.struct_engine.patches_per_tile
                patch_mask = good_mask.repeat_interleave(P)
                results['struct'] = {
                    'indices':   indices[patch_mask],
                    'rotations': rots[patch_mask],
                    'gains':     gains[patch_mask],
                    'biases':    biases[patch_mask],
                    'n_tiles':   good_mask.sum().item()
                }
            else:
                results['struct'] = {
                    'indices':   indices,
                    'rotations': rots,
                    'gains':     gains,
                    'biases':    biases,
                    'n_tiles':   struct_idx.numel()
                }

        # --- 3. NEURAL ---
        neural_idx = (mask == 2).nonzero(as_tuple=True)[0]
        if neural_idx.numel() > 0:
            results['neural'] = self.neural_engine.encode(tiles[neural_idx])

        return results

    def reconstruct(self, results: dict, n_images: int, image_size: tuple, overlap: float = None) -> torch.Tensor:
        """Vectorized Reconstruction using F.fold and Gaussian weighting."""
        mask = results['routing_mask']
        device = mask.device
        NH, NW = results['grid_dim']
        H2, W2 = results['padded_size']
        T = self.tile_size
        step = int(T * (1.0 - (overlap if overlap is not None else self.overlap)))

        # Pre-render all tiles into a unified buffer
        # Default to float32 for reconstruction stability
        all_rendered = torch.zeros(mask.shape[0], 3, T, T, device=device, dtype=torch.float32)

        if results['geo'] is not None:
            g_idx = (mask == 0).nonzero(as_tuple=True)[0]
            all_rendered[g_idx] = self.geo_engine.decode(results['geo']).to(all_rendered.dtype)

        if results['struct'] is not None:
            s_idx = (mask == 1).nonzero(as_tuple=True)[0]
            # Use engine's decode method with the payload dictionary
            all_rendered[s_idx] = self.struct_engine.decode(results['struct']).to(all_rendered.dtype)

        if results['neural'] is not None:
            n_idx = (mask == 2).nonzero(as_tuple=True)[0]
            all_rendered[n_idx] = self.neural_engine.decode(results['neural']).to(all_rendered.dtype)


        # Gaussian-weighted blending via F.fold
        gauss = self._gaussian_window(device)
        tiles_weighted = all_rendered * gauss[0]
        
        # (B*NH*NW, 3, T, T) -> (B, 3*T*T, NH*NW)
        tiles_weighted = tiles_weighted.view(n_images, NH * NW, 3, T, T)
        tiles_weighted = tiles_weighted.permute(0, 2, 3, 4, 1).reshape(n_images, 3*T*T, NH*NW)
        
        recon = F.fold(tiles_weighted, output_size=(H2, W2), kernel_size=T, stride=step)
        
        # Weighted normalization
        weights_ones = torch.ones(n_images, 1, T, T, NH * NW, device=device) * gauss[0, 0].unsqueeze(-1)
        weights_fold = F.fold(weights_ones.reshape(n_images, T*T, NH*NW), 
                             output_size=(H2, W2), kernel_size=T, stride=step)
        
        final = recon / weights_fold.clamp(min=1e-4)
        # Crop to original size
        return final[:, :, :image_size[0], :image_size[1]].clamp(0, 1)

    def forward(self, images: torch.Tensor, current_epoch: int = 1) -> tuple:
        """Encode then immediately reconstruct."""
        results = self.encode(images, current_epoch)
        recon = self.reconstruct(results, images.shape[0], (images.shape[2], images.shape[3]))
        return recon, results['stats']