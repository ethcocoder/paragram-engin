import torch
import torch.nn as nn
from src.core.complexity_mask import ComplexityMask
from src.engines.geometric.surface_fit import PolynomialSurfaceFitter
from src.engines.structural.dictionary import StructuralEngine
from src.engines.neural.lightweight import DetailEngine

class AetherOrchestrator(nn.Module):
    """
    The Hybrid Brain (The Switch): Coordinates the Complexity Mask and routes
    tiles to the optimal compression engine (Geometric, Structural, or Neural).
    
    Threshold Curriculum:
        Instead of a hard similarity threshold, the threshold ramps linearly
        from `curriculum_start` (0.70) to `curriculum_end` (0.90) over
        `curriculum_epochs` (10) epochs. This allows the Structural Engine
        to begin contributing early while its codebook is still immature.
    """
    def __init__(self, config=None, curriculum_start=0.70, curriculum_end=0.90, curriculum_epochs=10):
        super().__init__()
        self.complexity_mask = ComplexityMask()
        self.geometric_engine = PolynomialSurfaceFitter()
        self.structural_engine = StructuralEngine()
        self.neural_engine = DetailEngine()
        
        # Curriculum parameters (not nn.Parameters — these are scheduling constants)
        self.curriculum_start = curriculum_start
        self.curriculum_end = curriculum_end
        self.curriculum_epochs = curriculum_epochs
    
    def get_curriculum_threshold(self, current_epoch):
        """Compute the similarity threshold for the current epoch."""
        if current_epoch >= self.curriculum_epochs:
            return self.curriculum_end
        progress = current_epoch / max(self.curriculum_epochs, 1)
        return self.curriculum_start + progress * (self.curriculum_end - self.curriculum_start)
        
    def forward(self, tiles, sim_threshold=None, current_epoch=0):
        """
        Routes tiles to engines based on Complexity Mask with a Neural Fallback.
        If structural match similarity is < threshold, it falls back to Neural.
        
        Args:
            tiles: (B, 3, 128, 128) input tile batch.
            sim_threshold: explicit override; if None, uses curriculum schedule.
            current_epoch: used to compute curriculum threshold when sim_threshold is None.
        """
        B = tiles.shape[0]
        
        # Resolve threshold via curriculum schedule
        if sim_threshold is None:
            sim_threshold = self.get_curriculum_threshold(current_epoch)
        
        mask = self.complexity_mask(tiles) # (B, 1, 1, 1)
        
        results = {
            'mask': mask,
            'geometric': None,
            'structural': None,
            'neural': None
        }
        
        # 1. Geometric Path
        geom_indices = (mask == 0).view(-1)
        if geom_indices.any():
            results['geometric'] = self.geometric_engine.fit(tiles[geom_indices])
            
        # 2. Structural Path with Fallback
        struct_indices = (mask == 1).view(-1)
        if struct_indices.any():
            # Initial encode to check similarity
            struct_out = self.structural_engine.encode(tiles[struct_indices])
            indices, rots, gains, biases, sims = struct_out
            
            # Identify weak matches (Tile-level decision)
            # sims has shape (N_tiles * 16,) -> reshape to (N_tiles, 16)
            sims_per_tile = sims.view(-1, 16)
            # If ANY patch in the tile is weak, re-route the entire tile
            weak_tile_mask = (sims_per_tile.min(dim=1)[0] < sim_threshold)
            
            if weak_tile_mask.any():
                # Re-route weak tiles to Neural mode (2)
                global_struct_indices = torch.where(struct_indices)[0]
                global_weak_indices = global_struct_indices[weak_tile_mask]
                mask[global_weak_indices] = 2
                
                # Filter structural results for only strong tiles
                strong_tile_mask = ~weak_tile_mask
                if strong_tile_mask.any():
                    # We need to filter the patch-level data too (16 patches per tile)
                    # Create a patch-level mask from the tile-level mask
                    strong_patch_mask = strong_tile_mask.repeat_interleave(16)
                    results['structural'] = (
                        indices[strong_patch_mask], rots[strong_patch_mask], 
                        gains[strong_patch_mask], biases[strong_patch_mask], sims[strong_patch_mask]
                    )
            else:
                results['structural'] = struct_out
            
        # 3. Neural Path (including Fallbacks)
        neural_indices = (mask == 2).view(-1)
        if neural_indices.any():
            results['neural'] = self.neural_engine.encode(tiles[neural_indices])
            
        return results

    def reconstruct(self, results, B, image_size=256, overlap=16):
        """
        Reassembles an image from hybrid results using Overlap-Aware Stitching.
        Uses F.fold to blend tiles with a Gaussian-like weighting mask.
        """
        mask = results['mask']
        device = mask.device
        
        target_dtype = torch.float32
        for key in ['neural', 'geometric']:
            if results[key] is not None:
                target_dtype = results[key].dtype
                break
            
        tile_size = 128
        stride = tile_size - overlap
        
        # 1. Create a Weighting Mask (Cosine/Gaussian taper)
        # Smoothly fades out towards the edges
        taper = torch.hann_window(tile_size, periodic=False).view(1, tile_size)
        weight_mask = (taper.T @ taper).to(device).to(target_dtype) # (128, 128)
        
        # 2. Accumulators for Fold
        # We process image by image to avoid huge memory spikes in Fold
        # B_img is calculated from the total number of tiles B
        # tiles_per_side = (image_size - tile_size) // stride + 1
        # B = B_img * (tiles_per_side**2)
        tiles_per_side = (image_size - tile_size) // stride + 1
        B_img = B // (tiles_per_side**2)
        
        # We'll render all tiles first
        all_rendered = torch.zeros(B, 3, 128, 128, device=device, dtype=target_dtype)
        
        # Restore Geometric
        if results['geometric'] is not None:
            geom_indices = (mask == 0).view(-1)
            all_rendered[geom_indices] = self.geometric_engine.render(results['geometric']).to(target_dtype)
            
        # Restore Structural
        if results['structural'] is not None:
            struct_indices = (mask == 1).view(-1)
            indices, rotations, gains, biases, _ = results['structural']
            B_struct = struct_indices.sum().item()
            render_out = self.structural_engine.render(
                indices, rotations, gains, biases, int(B_struct)
            )
            all_rendered[struct_indices] = render_out.to(target_dtype)
            
        # Restore Neural
        if results['neural'] is not None:
            neural_indices = (mask == 2).view(-1)
            all_rendered[neural_indices] = self.neural_engine.decode(results['neural']).to(target_dtype)
            
        # 3. Blending with Fold
        # Apply weighting mask to all rendered tiles
        # Clamp to prevent extreme values from exploding
        all_rendered = torch.clamp(all_rendered, -10.0, 10.0)
        # Replace any residual NaNs with zeros (Artifact Protection)
        all_rendered = torch.where(torch.isnan(all_rendered), torch.zeros_like(all_rendered), all_rendered)
        all_rendered = all_rendered * weight_mask.view(1, 1, 128, 128)
        
        # Reshape for fold: (B_img, 3 * 128 * 128, num_tiles)
        num_tiles_per_img = tiles_per_side**2
        all_rendered = all_rendered.view(B_img, num_tiles_per_img, 3, 128, 128).permute(0, 2, 3, 4, 1).reshape(B_img, 3 * 128 * 128, num_tiles_per_img)
        
        # Output accumulation
        combined = torch.nn.functional.fold(
            all_rendered, 
            output_size=(image_size, image_size), 
            kernel_size=(128, 128), 
            stride=(stride, stride)
        )
        
        # Weight accumulation (to normalize the overlap)
        ones = torch.ones(B_img, num_tiles_per_img, 1, 128, 128, device=device, dtype=target_dtype) * weight_mask.view(1, 1, 1, 128, 128)
        ones = ones.permute(0, 2, 3, 4, 1).reshape(B_img, 1 * 128 * 128, num_tiles_per_img)
        weight_sum = torch.nn.functional.fold(
            ones, 
            output_size=(image_size, image_size), 
            kernel_size=(128, 128), 
            stride=(stride, stride)
        )
        
        # Use epsilon = 1e-6 in the division to prevent NaN (Artifact Protection)
        return torch.clamp(combined / (weight_sum + 1e-6), 0.0, 1.0)
