import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import numpy as np

class ComplexityMask(nn.Module):
    def __init__(self, threshold_low=0.01, threshold_periodicity=50.0):
        super().__init__()
        # Learnable thresholds
        self.threshold_low = nn.Parameter(torch.tensor([threshold_low]))
        self.threshold_periodicity = nn.Parameter(torch.tensor([threshold_periodicity]))
        
        # Laplacian Kernel for variance/energy calculation
        kernel = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('laplacian_kernel', kernel)

    def forward(self, x):
        """
        Analyzes batch of tiles and returns a Tri-State Mask.
        Args:
            x (Tensor): Input image tiles [B, 3, 128, 128]
        Returns:
            Tensor: Tri-state mask [B, 1] with values {0, 1, 2}
        """
        B, C, H, W = x.shape
        
        # Convert to grayscale for analysis
        x_gray = x.mean(dim=1, keepdim=True)
        
        # --- Level 1: Entropy Check (Vacuum vs Complex) ---
        # Calculate Laplacian variance per tile
        laplacian = F.conv2d(x_gray, self.laplacian_kernel, padding=1)
        variance = torch.var(laplacian, dim=(2, 3)) # [B, 1]
        
        # Initial mask: 0 for low variance, 2 for high (will refine to 1 later)
        mask = torch.where(variance < self.threshold_low, 
                          torch.zeros_like(variance), 
                          torch.full_like(variance, 2.0))
        
        # --- Level 2: Periodicity Check (Texture vs Detail) ---
        # Only analyze tiles that are NOT State 0
        complex_indices = (mask > 0).view(-1)
        
        if complex_indices.any():
            x_complex = x_gray[complex_indices]
            
            # 2D Real FFT
            # Shifted FFT to put DC in center? Actually rfft2 is enough for power spectrum
            fft = torch.fft.rfft2(x_complex, norm='ortho')
            power_spectrum = torch.abs(fft)
            
            # Calculate Periodicity Score
            # We look for dominant peaks relative to the mean power (excluding DC)
            # Flatten spatial dims of spectrum
            power_flat = power_spectrum.view(x_complex.shape[0], -1)
            
            # Remove DC component (first element)
            power_no_dc = power_flat[:, 1:]
            
            max_power = torch.max(power_no_dc, dim=1)[0]
            mean_power = torch.mean(power_no_dc, dim=1)
            
            # Periodicity Score: Ratio of Max Power to Mean Power
            # High ratio indicates a strong repeating pattern (peakiness)
            periodicity_score = max_power / (mean_power + 1e-8)
            
            # Refine mask: If periodicity > threshold, set to 1 (Texture)
            is_texture = (periodicity_score > self.threshold_periodicity).to(mask.dtype)
            
            # Map back to original indices
            # complex_indices is a boolean mask of shape [B]
            complex_mask = mask[complex_indices]
            # If is_texture is 1, we want result to be 1. If 0, keep as 2.
            # Logic: result = 1 if is_texture else 2
            refined_states = torch.where(is_texture.unsqueeze(1) > 0, 
                                        torch.ones_like(complex_mask), 
                                        torch.full_like(complex_mask, 2.0))
            
            mask[complex_indices] = refined_states

        return mask.long()

    @staticmethod
    def visualize_mask(image, mask, tile_size=128):
        """
        Overlays tri-state colors on the original image for debugging.
        Blue (0): Vacuum
        Green (1): Texture
        Red (2): Detail
        
        Args:
            image (Tensor): Full image [3, H, W]
            mask (Tensor): Mask per tile [N_tiles, 1]
        """
        import matplotlib.pyplot as plt
        
        C, H, W = image.shape
        num_tiles_h = H // tile_size
        num_tiles_w = W // tile_size
        
        # Reshape mask to 2D grid
        mask_grid = mask.view(num_tiles_h, num_tiles_w).cpu().numpy()
        
        # Create color overlay
        overlay = np.zeros((H, W, 3))
        
        colors = [
            [0, 0, 1], # 0: Blue (Vacuum)
            [0, 1, 0], # 1: Green (Texture)
            [1, 0, 0]  # 2: Red (Detail)
        ]
        
        for i in range(num_tiles_h):
            for j in range(num_tiles_w):
                state = int(mask_grid[i, j])
                color = colors[state]
                y1, y2 = i * tile_size, (i + 1) * tile_size
                x1, x2 = j * tile_size, (j + 1) * tile_size
                overlay[y1:y2, x1:x2] = color
                
        # Alpha blending
        img_np = image.permute(1, 2, 0).cpu().numpy()
        # Normalize image to 0-1 if it isn't
        if img_np.max() > 1.0: img_np /= 255.0
        
        blended = img_np * 0.6 + overlay * 0.4
        
        return blended
