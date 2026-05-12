import torch
import torch.nn as nn
import torch.nn.functional as F

class TextureCodebook(nn.Module):
    """
    Holds a dictionary of visual patterns (visual words).
    Shape: (num_entries, 3, 32, 32)
    """
    def __init__(self, num_entries=512, patch_size=32):
        super().__init__()
        self.num_entries = num_entries
        self.patch_size = patch_size
        
        # Initialize codebook with Gaussian distribution
        # In a real scenario, this would be initialized via K-means on a dataset
        self.codebook = nn.Parameter(torch.randn(num_entries, 3, patch_size, patch_size) * 0.1)

class StructuralEngine(nn.Module):
    """
    The Pattern Memory: Matches State 1 (Texture) tiles against the codebook.
    Supports 4-way rotation and affine correction (Gain/Bias).
    """
    def __init__(self, num_entries=512, patch_size=32, fallback_threshold=0.80):
        super().__init__()
        self.patch_size = patch_size
        self.fallback_threshold = fallback_threshold
        self.codebook_module = TextureCodebook(num_entries, patch_size)
        
    def _subdivide(self, tiles):
        """(B, 3, 128, 128) -> (B * 16, 3, 32, 32)"""
        B, C, H, W = tiles.shape
        # Unfold extracts sliding patches. With stride=patch_size, we get non-overlapping patches.
        patches = tiles.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        # patches shape: (B, 3, 4, 4, 32, 32)
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(-1, C, self.patch_size, self.patch_size)
        return patches

    def encode(self, tiles):
        """
        Full structural encoding pipeline.
        Returns indices, params, and the maximum similarity scores.
        """
        patches = self._subdivide(tiles)
        return self._match_vectorized(patches)

    def _match_vectorized(self, patches):
        """
        Matches patches against codebook with 4 rotations.
        Returns:
        indices (B*16), rotations (B*16), gains (B*16), biases (B*16), max_sims (B*16)
        """
        # ... (implementation same as before but returning max_sim)
        cb = self.codebook_module.codebook
        cb_rots = torch.stack([
            cb,
            torch.rot90(cb, 1, [2, 3]),
            torch.rot90(cb, 2, [2, 3]),
            torch.rot90(cb, 3, [2, 3])
        ], dim=0)
        
        N_rots, N_entries, C, H, W = cb_rots.shape
        num_patches = patches.shape[0]
        
        patches_flat = patches.reshape(num_patches, -1)
        cb_rots_flat = cb_rots.reshape(N_rots * N_entries, -1)
        
        patches_norm = F.normalize(patches_flat, p=2, dim=1)
        cb_rots_norm = F.normalize(cb_rots_flat, p=2, dim=1)
        
        sim = torch.matmul(patches_norm, cb_rots_norm.t())
        max_sim, best_idx_flat = torch.max(sim, dim=1)
        
        rot_indices = best_idx_flat // N_entries
        entry_indices = best_idx_flat % N_entries
        
        best_cb_patches = cb_rots[rot_indices, entry_indices]
        
        p_mean = patches.mean(dim=(1,2,3), keepdim=True)
        e_mean = best_cb_patches.mean(dim=(1,2,3), keepdim=True)
        
        p_centered = patches - p_mean
        e_centered = best_cb_patches - e_mean
        
        gain = (p_centered * e_centered).sum(dim=(1,2,3)) / ((e_centered**2).sum(dim=(1,2,3)) + 1e-8)
        gain = torch.clamp(gain, 0.1, 10.0)
        bias = p_mean.squeeze() - gain * e_mean.squeeze()
        
        return entry_indices, rot_indices, gain, bias, max_sim

    def render(self, indices, rotations, gains, biases, B):
        """
        Reconstructs tiles from codebook indices and parameters.
        """
        cb = self.codebook_module.codebook
        N_entries, C, H, W = cb.shape
        
        # Retrieve base patches
        patches = cb[indices] # (P, 3, 32, 32)
        
        # Apply 4-way rotations (vectorized is tricky, using a loop over 4 rotations is faster/cleaner here)
        final_patches = torch.zeros_like(patches)
        for r in range(4):
            mask = (rotations == r)
            if mask.any():
                final_patches[mask] = torch.rot90(patches[mask], r, [2, 3])
                
        # Apply Gain and Bias
        # gains: (P), biases: (P)
        final_patches = final_patches * gains.view(-1, 1, 1, 1) + biases.view(-1, 1, 1, 1)
        
        # Reshape and stitch back into (B, 3, 128, 128)
        # final_patches: (B*16, 3, 32, 32)
        P = final_patches.shape[0]
        tiles = final_patches.reshape(B, 4, 4, 3, 32, 32)
        tiles = tiles.permute(0, 3, 1, 4, 2, 5).reshape(B, 3, 128, 128)
        
        return tiles
