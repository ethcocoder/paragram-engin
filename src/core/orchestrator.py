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

    def reconstruct(self, results, B):
        """
        Reassembles an image from the hybrid results.
        """
        mask = results['mask']
        device = mask.device
        
        # We'll use the dtype of the first available engine output, or default to float32
        target_dtype = torch.float32
        for key in ['neural', 'geometric']:
            if results[key] is not None:
                target_dtype = results[key].dtype
                break
            
        reconstructed_tiles = torch.zeros(B, 3, 128, 128, device=device, dtype=target_dtype)
        
        # Restore Geometric
        if results['geometric'] is not None:
            geom_indices = (mask == 0).view(-1)
            reconstructed_tiles[geom_indices] = self.geometric_engine.render(results['geometric']).to(target_dtype)
            
        # Restore Structural
        if results['structural'] is not None:
            struct_indices = (mask == 1).view(-1)
            indices, rotations, gains, biases, _ = results['structural']
            B_struct = struct_indices.sum().item()
            render_out = self.structural_engine.render(
                indices, rotations, gains, biases, int(B_struct)
            )
            reconstructed_tiles[struct_indices] = render_out.to(target_dtype)
            
        # Restore Neural
        if results['neural'] is not None:
            neural_indices = (mask == 2).view(-1)
            reconstructed_tiles[neural_indices] = self.neural_engine.decode(results['neural']).to(target_dtype)
            
        return reconstructed_tiles
