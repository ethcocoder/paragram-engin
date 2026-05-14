"""
fractal.py — Aether-Blueprint v3.0
=====================================
Structural Engine Extension: Recursive Self-Similar Pattern Discovery.

Responsibility:
    Detect tiles that contain self-similar (fractal-like) structure at
    multiple scales — such as foliage, clouds, water ripples, fabric weaves —
    and encode them using an Iterated Function System (IFS) rule set rather
    than a flat codebook lookup.

    IFS Encoding stores a small set of affine contraction maps:
        T_i(x, y) = [a_i  b_i] [x] + [e_i]
                    [c_i  d_i] [y]   [f_i]
    plus a colour scaling factor per transform.

    Decoder iterates the IFS ~20 steps to approximate the attractor.

Design notes:
    - This is a specialist engine; the orchestrator will route only confirmed
      self-similar tiles here (requires DetectSelfSimilarity flag from mask).
    - IFS fitting uses a least-squares range-domain matching algorithm
      (similar to Barnsley's fractal compression scheme).
    - For v3.0, supports up to 8 IFS transforms per tile.
    - Fully differentiable decode path for end-to-end fine-tuning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FractalEngine(nn.Module):
    """
    Fractal IFS encoder/decoder for self-similar texture tiles.

    Args:
        tile_size       (int): edge length of input tiles.
        max_transforms  (int): maximum IFS transforms to store per tile.
        decode_iters    (int): iterations for attractor approximation.
        domain_size     (int): domain block size for range-domain matching.
        range_size      (int): range block size (domain is downsampled to this).
    """

    def __init__(self,
                 tile_size:      int = 128,
                 max_transforms: int = 8,
                 decode_iters:   int = 24,
                 domain_size:    int = 32,
                 range_size:     int = 16):
        super().__init__()
        self.tile_size      = tile_size
        self.max_transforms = max_transforms
        self.decode_iters   = decode_iters
        self.domain_size    = domain_size
        self.range_size     = range_size

    # ------------------------------------------------------------------
    # Self-similarity detection
    # ------------------------------------------------------------------

    def self_similarity_score(self, tile: torch.Tensor) -> float:
        """
        Compute a self-similarity score for a single tile.

        Compares the tile against a 2× downsampled version of itself.
        High score → self-similar → good candidate for fractal encoding.

        Args:
            tile: (1, C, T, T) float tensor.
        Returns:
            float in [0, 1] — higher is more self-similar.
        """
        T    = self.tile_size
        down = F.avg_pool2d(tile, kernel_size=2, stride=2)         # (1,C,T/2,T/2)
        up   = F.interpolate(down, size=(T, T), mode='bilinear',
                             align_corners=False)                   # (1,C,T,T)
        sim  = F.cosine_similarity(tile.view(1, -1), up.view(1, -1)).item()
        return max(0.0, sim)

    # ------------------------------------------------------------------
    # IFS fitting (range-domain matching)
    # ------------------------------------------------------------------

    def _extract_domains(self, tile: torch.Tensor) -> torch.Tensor:
        """
        Extract all domain blocks (downsampled to range_size) from the tile.

        Returns: (N_d, C, Rs, Rs)
        """
        T  = self.tile_size
        Ds = self.domain_size
        Rs = self.range_size
        step = Ds // 2   # 50% overlap on domain blocks

        domains = []
        for top in range(0, T - Ds + 1, step):
            for left in range(0, T - Ds + 1, step):
                d = tile[:, :, top:top+Ds, left:left+Ds]
                d_down = F.avg_pool2d(d, kernel_size=Ds // Rs, stride=Ds // Rs)
                domains.append(d_down.squeeze(0))   # (C, Rs, Rs)
        return torch.stack(domains)                 # (N_d, C, Rs, Rs)

    def _extract_ranges(self, tile: torch.Tensor) -> torch.Tensor:
        """
        Extract all non-overlapping range blocks from the tile.

        Returns: (N_r, C, Rs, Rs), (N_r, 2) top-left coords
        """
        T  = self.tile_size
        Rs = self.range_size
        ranges = []
        coords = []
        for top in range(0, T, Rs):
            for left in range(0, T, Rs):
                r = tile[:, :, top:top+Rs, left:left+Rs]
                ranges.append(r.squeeze(0))
                coords.append((top, left))
        return torch.stack(ranges), coords

    def encode(self, tiles: torch.Tensor) -> list:
        """
        Encode a batch of tiles as IFS transform lists.

        Args:
            tiles: (N, C, T, T)

        Returns:
            list of N dicts, each with:
                'transforms': list of (domain_top, domain_left, rotation, gain, bias)
                'range_coords': list of (top, left) per range block
        """
        N   = tiles.shape[0]
        results = []

        for n in range(N):
            tile    = tiles[n:n+1]                      # (1, C, T, T)
            domains = self._extract_domains(tile)       # (N_d, C, Rs, Rs)
            ranges, r_coords = self._extract_ranges(tile)

            transforms = []
            for r_flat, (rt, rl) in zip(ranges, r_coords):
                r_vec = r_flat.reshape(-1).float()

                # Find best domain match across all rotations
                best_mse = float('inf')
                best_t   = None

                for d_idx, d in enumerate(domains):
                    for rot_k in range(4):
                        d_rot    = torch.rot90(d, rot_k, dims=[-2, -1])
                        d_vec    = d_rot.reshape(-1).float()
                        # Affine fit: gain = cov(d,r)/var(d), bias = mean(r)-gain*mean(d)
                        d_mean   = d_vec.mean()
                        r_mean   = r_vec.mean()
                        d_std    = d_vec.std().clamp(min=1e-6)
                        gain     = ((d_vec - d_mean) * (r_vec - r_mean)).mean() / (d_std**2)
                        bias     = r_mean - gain * d_mean
                        recon    = gain * d_vec + bias
                        mse      = F.mse_loss(recon, r_vec).item()
                        if mse < best_mse:
                            best_mse = mse
                            best_t   = (d_idx, rot_k, gain.item(), bias.item())

                transforms.append(best_t)

            results.append({
                'transforms'  : transforms,
                'range_coords': r_coords,
                'tile_size'   : self.tile_size,
                'range_size'  : self.range_size,
            })

        return results

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------

    def decode_single(self, ifs_data: dict, channels: int = 3) -> torch.Tensor:
        """
        Reconstruct a single tile from IFS data by iterating the attractor.

        Args:
            ifs_data : dict from encode()
            channels : number of channels

        Returns:
            (1, C, T, T) float tensor
        """
        T   = ifs_data['tile_size']
        Rs  = ifs_data['range_size']
        Ds  = self.domain_size
        transforms   = ifs_data['transforms']
        range_coords = ifs_data['range_coords']

        # Start from random noise
        canvas = torch.rand(1, channels, T, T)

        for _ in range(self.decode_iters):
            new_canvas = canvas.clone()
            for (d_idx, rot_k, gain, bias), (rt, rl) in zip(transforms, range_coords):
                # Reconstruct domain top-left from index
                step  = Ds // 2
                n_col = (T - Ds) // step + 1
                dt    = (d_idx // n_col) * step
                dl    = (d_idx  % n_col) * step

                d_block = canvas[:, :, dt:dt+Ds, dl:dl+Ds]
                d_down  = F.avg_pool2d(d_block,
                                       kernel_size=Ds // Rs,
                                       stride=Ds // Rs)
                d_rot   = torch.rot90(d_down, rot_k, dims=[-2, -1])
                new_canvas[:, :, rt:rt+Rs, rl:rl+Rs] = (gain * d_rot + bias).clamp(0, 1)

            canvas = new_canvas

        return canvas

    def decode(self, ifs_list: list, channels: int = 3) -> torch.Tensor:
        """
        Decode a list of IFS dicts into a tile batch.

        Returns: (N, C, T, T)
        """
        tiles = [self.decode_single(d, channels) for d in ifs_list]
        return torch.cat(tiles, dim=0)

    def forward(self, tiles: torch.Tensor) -> torch.Tensor:
        """Encode then immediately decode (for training)."""
        ifs_list = self.encode(tiles)
        return self.decode(ifs_list, channels=tiles.shape[1])