"""
dictionary.py — Aether-Blueprint v3.0
=======================================
Structural Engine: Pattern Memory Codebook.
 
Responsibility:
    Encode repeating-texture tiles by matching 32×32 sub-patches against a
    learned 512-entry codebook.  Each match is stored as:
        (codebook_index, rotation_id ∈ {0,1,2,3}, gain, bias)
    totalling 7 bytes per patch vs. 32×32×3 = 3,072 bytes — a 439× ratio.
 
    At decode time, the codebook entry is rotated and affine-corrected to
    reproduce the original patch.
 
Architecture:
    - Codebook: nn.Embedding(512, 32*32*3) — learned during Stage 1.
    - Matching: L2 nearest-neighbour over the flattened patch × codebook.
                4-way rotation is tried; best match wins.
    - Affine correction: per-patch gain and bias computed analytically
                         from matched vs. target patch statistics.
 
Patch-vs-tile indexing fix (from Stage 1 bug):
    A 128×128 tile contains (128/32)² = 16 patches.
    Similarity scores are (N_tiles × 16,) — they must be reshaped to
    (N_tiles, 16) and reduced with min() before deciding neural fallback.
    This class now returns per-tile min similarity for the orchestrator.
"""
 
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
 
 
class StructuralEngine(nn.Module):
    """
    Dictionary-based texture codebook engine.
 
    Args:
        codebook_size (int): number of codebook entries (default 512)
        patch_size    (int): sub-patch edge length (default 32)
        tile_size     (int): full tile edge length (default 128)
    """
 
    ROTATIONS = [0, 1, 2, 3]   # 0°, 90°, 180°, 270°
 
    def __init__(self,
                 codebook_size: int = 512,
                 patch_size:    int = 32,
                 tile_size:     int = 128):
        super().__init__()
        self.codebook_size = codebook_size
        self.patch_size    = patch_size
        self.tile_size     = tile_size
        self.patches_per_tile = (tile_size // patch_size) ** 2
 
        # Codebook: each entry is a flat patch (P*P*C)
        self.codebook = nn.Embedding(codebook_size, patch_size * patch_size * 3)
        nn.init.normal_(self.codebook.weight, mean=0.5, std=0.1)
 
    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
 
    def _extract_patches(self, tiles: torch.Tensor) -> torch.Tensor:
        """
        Extract non-overlapping patches from tiles.
 
        Args:
            tiles: (N, C, T, T)
        Returns:
            patches: (N * P, C, Ps, Ps)  where P = patches_per_tile
        """
        N, C, T, _ = tiles.shape
        Ps = self.patch_size
        P  = T // Ps
        # unfold → (N, C, P, P, Ps, Ps) → (N*P², C, Ps, Ps)
        patches = tiles.unfold(2, Ps, Ps).unfold(3, Ps, Ps)
        patches = patches.contiguous().view(N, C, P, P, Ps, Ps)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.view(N * P * P, C, Ps, Ps)
        return patches
 
    @staticmethod
    def _rotate_k(x: torch.Tensor, k: int) -> torch.Tensor:
        """Rotate tensor (C, Ps, Ps) by k*90 degrees."""
        if k == 0:
            return x
        return torch.rot90(x, k, dims=[-2, -1])
 
    def _affine_correct(self,
                        src: torch.Tensor,
                        tgt: torch.Tensor) -> tuple:
        """
        Compute per-patch gain and bias so that  gain * src + bias ≈ tgt.
        Returns (gain, bias) as scalars per patch.
        """
        # Flatten spatial dims
        s = src.reshape(src.shape[0], -1).float()   # (N, D)
        t = tgt.reshape(tgt.shape[0], -1).float()
 
        s_mean = s.mean(dim=1, keepdim=True)
        t_mean = t.mean(dim=1, keepdim=True)
        s_std  = s.std(dim=1, keepdim=True).clamp(min=1e-6)
        t_std  = t.std(dim=1, keepdim=True).clamp(min=1e-6)
 
        gain = (t_std / s_std).squeeze(1)           # (N,)
        bias = (t_mean - gain.unsqueeze(1) * s_mean).squeeze(1)  # (N,)
        return gain, bias
 
    # ------------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------------
 
    def encode(self, tiles: torch.Tensor) -> dict:
        """
        Encode a batch of tiles to codebook indices + affine params.
 
        Args:
            tiles: (N, C, T, T) float tensor.
 
        Returns:
            dict with:
                'indices'       : (N*P,) LongTensor — codebook index per patch
                'rotations'     : (N*P,) LongTensor — rotation id
                'gains'         : (N*P,) FloatTensor
                'biases'        : (N*P,) FloatTensor
                'similarities'  : (N, P) FloatTensor — min sim per tile (for OOL check)
                'n_tiles'       : int
        """
        N  = tiles.shape[0]
        Ps = self.patch_size
        P  = self.patches_per_tile
 
        patches   = self._extract_patches(tiles)    # (N*P, C, Ps, Ps)
        NP        = patches.shape[0]
        flat_p    = patches.view(NP, -1).float()    # (N*P, D)  D=C*Ps*Ps
 
        # Codebook entries - force float32 for matching precision
        cb        = self.codebook.weight.float()      # (K, D)
        cb_norm   = F.normalize(cb, dim=1)
        flat_norm = F.normalize(flat_p, dim=1)
 
        # Try all 4 rotations, keep best
        D = Ps * Ps * 3
        # Initialize best_sim as float32 to match matching results
        best_sim  = torch.full((NP,), -1.0, device=tiles.device, dtype=torch.float32)
        best_idx  = torch.zeros(NP, dtype=torch.long, device=tiles.device)
        best_rot  = torch.zeros(NP, dtype=torch.long, device=tiles.device)
 
        # FORCE DISABLE AUTOCAST for the matching loop to ensure float32
        with torch.amp.autocast('cuda', enabled=False):
            for rot_k in self.ROTATIONS:
                rot_patches  = torch.stack([self._rotate_k(patches[i], rot_k)
                                            for i in range(NP)])   # (NP, C, Ps, Ps)
                # Ensure rot_flat is float32
                rot_flat     = F.normalize(rot_patches.view(NP, -1).float(), dim=1)
                # Ensure matmul is float32
                sims         = rot_flat @ cb_norm.t()               # (NP, K)
                top_sim, top_idx = sims.max(dim=1)                  # (NP,)
                top_sim = top_sim.float() # ENSURE float32 for assignment
                better = top_sim > best_sim
                best_sim[better] = top_sim[better]
                best_idx[better] = top_idx[better]
                best_rot[better] = rot_k
 
        # Affine correction using best-matching (rotated) codebook entry
        matched_cb  = self.codebook(best_idx)                   # (NP, D)
        matched_tiled = torch.zeros_like(matched_cb)
        for rot_k in self.ROTATIONS:
            mask = best_rot == rot_k
            if mask.any():
                # rotate the codebook patch
                cb_patches = matched_cb[mask].view(-1, 3, Ps, Ps)
                rot_cb     = torch.stack([self._rotate_k(cb_patches[i], rot_k)
                                          for i in range(cb_patches.shape[0])])
                matched_tiled[mask] = rot_cb.view(-1, D)
 
        gains, biases = self._affine_correct(matched_tiled, flat_p)

        # Reshape similarities to (N, P) for tile-level min check
        sim_per_tile = best_sim.view(N, P)          # (N, P)

        # Cast back to input dtype for gains/biases to maintain consistency in mixed precision
        orig_dtype = tiles.dtype
        return {
            'indices'     : best_idx,               # (N*P,)
            'rotations'   : best_rot,               # (N*P,)
            'gains'       : gains.to(orig_dtype),    # (N*P,)
            'biases'      : biases.to(orig_dtype),   # (N*P,)
            'similarities': sim_per_tile,           # (N, P) float32
            'n_tiles'     : N,
        }

 
    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------
 
    def decode(self, encoded: dict) -> torch.Tensor:
        """
        Reconstruct tiles from codebook indices and affine params.
 
        Args:
            encoded: dict from encode()
 
        Returns:
            tiles: (N, C, T, T) float tensor.
        """
        indices   = encoded['indices']
        rotations = encoded['rotations']
        gains     = encoded['gains']
        biases    = encoded['biases']
        N         = encoded['n_tiles']
        Ps        = self.patch_size
        T         = self.tile_size
        P_edge    = T // Ps                          # patches per row/col
        P         = self.patches_per_tile
        NP        = N * P
        device    = indices.device
 
        # Fetch and rotate codebook entries
        cb_entries = self.codebook(indices)          # (NP, D)
        reconstructed = torch.zeros_like(cb_entries)
        for rot_k in self.ROTATIONS:
            mask = rotations == rot_k
            if mask.any():
                cp = cb_entries[mask].view(-1, 3, Ps, Ps)
                rp = torch.stack([self._rotate_k(cp[i], rot_k)
                                  for i in range(cp.shape[0])])
                reconstructed[mask] = rp.view(-1, cb_entries.shape[1])
 
        # Apply affine correction
        g = gains.view(NP, 1)
        b = biases.view(NP, 1)
        reconstructed = g * reconstructed + b       # (NP, D)
 
        # Reassemble patches into tiles
        patches = reconstructed.view(NP, 3, Ps, Ps)
        patches = patches.view(N, P_edge, P_edge, 3, Ps, Ps)
        patches = patches.permute(0, 3, 1, 4, 2, 5).contiguous()
        tiles   = patches.view(N, 3, T, T)
        return tiles.clamp(0.0, 1.0)
 