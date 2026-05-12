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
    """
    def __init__(self, config=None):
        super().__init__()
        self.complexity_mask = ComplexityMask()
        self.geometric_engine = PolynomialSurfaceFitter()
        self.structural_engine = StructuralEngine()
        self.neural_engine = DetailEngine()
        
    def forward(self, tiles):
        """
        Routes a batch of tiles to the appropriate engines.
        Args:
            tiles (Tensor): (B, 3, 128, 128)
        Returns:
            dict: Containing coefficients, indices, and latents per mode.
        """
        B, C, H, W = tiles.shape
        
        # 1. Generate Complexity Mask
        # mask shape: (B, 1) with values {0, 1, 2}
        mask = self.complexity_mask(tiles)
        
        results = {
            'mask': mask,
            'geometric': None, # Coeffs
            'structural': None, # (Indices, Rots, Gains, Biases)
            'neural': None, # Latents
        }
        
        # 2. Route to Geometric Engine (State 0)
        geom_indices = (mask == 0).view(-1)
        if geom_indices.any():
            geom_tiles = tiles[geom_indices]
            results['geometric'] = self.geometric_engine.fit(geom_tiles)
            
        # 3. Route to Structural Engine (State 1)
        struct_indices = (mask == 1).view(-1)
        if struct_indices.any():
            struct_tiles = tiles[struct_indices]
            # Structural engine returns: indices, rotations, gains, biases, fallback_mask
            struct_data = self.structural_engine(struct_tiles)
            
            # Handle Fallback from Structural to Neural
            # fallback_mask is (B_struct * 16)
            # For simplicity in this orchestrator, we'll keep them as structural 
            # and let the trainer/packer handle fallback if needed.
            results['structural'] = struct_data
            
        # 4. Route to Neural Engine (State 2)
        neural_indices = (mask == 2).view(-1)
        if neural_indices.any():
            neural_tiles = tiles[neural_indices]
            results['neural'] = self.neural_engine.encode(neural_tiles)
            
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
        
        # Use larger epsilon and clamp final image
        return torch.clamp(combined / (weight_sum + 1e-4), 0.0, 1.0)
