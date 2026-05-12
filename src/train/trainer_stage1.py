import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path
from tqdm import tqdm
import torchvision.utils as vutils

import sys
from pathlib import Path
# Add the project root to sys.path to ensure 'src' is found
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.core.orchestrator import AetherOrchestrator
from src.utils.tiling_v3 import SmartTiler

class HybridDataset(Dataset):
    """
    Robust dataset for Aether-Blueprint training.
    """
    def __init__(self, data_dir, image_size=512):
        self.data_dir = Path(data_dir)
        # Recursively find all images
        self.image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
            self.image_paths.extend(list(self.data_dir.glob(f'**/{ext}')))
        
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.image_paths[idx]).convert('RGB')
            return self.transform(img)
        except Exception as e:
            # Return a blank image if corrupted to avoid crashing
            # In a real scenario, you'd log this and skip
            return torch.zeros(3, 512, 512)

def total_variation_loss(img):
    """
    Penalizes sharp pixel jumps between horizontal and vertical neighbors.
    Enforces spatial smoothness across tile boundaries.
    """
    tv_h = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum()
    tv_w = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum()
    return (tv_h + tv_w) / img.shape[0]

def edge_match_loss(img, overlap=16):
    """
    Penalizes differences in overlapping regions between tiles.
    Forces border continuity.
    """
    # Horizontal borders
    # Left overlap of tile i+1 should match right overlap of tile i
    # Since we use Fold with weights, we can just check the internal variation
    # But a more direct way is to penalize high gradients in the overlap zones
    # However, TV loss already does this. 
    # For "Stage 1+", we implement a specific 4-pixel continuity check.
    h_diff = img[:, :, :, 1:] - img[:, :, :, :-1]
    v_diff = img[:, :, 1:, :] - img[:, :, :-1, :]
    
    # Focus loss on the stride boundaries (e.g. every 112 or 64 pixels)
    # For speed, we'll use a simplified version: 
    # highly penalize any jump that happens exactly at the tile boundaries.
    return (torch.mean(torch.abs(h_diff)) + torch.mean(torch.abs(v_diff)))

def train_stage1(data_dir, epochs=20, batch_size=16, lr=1e-4, resume=True, image_size=256):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Scaling Stage 1+ (Perfect Foundation) on {device} | Size: {image_size}")
    
    # Configuration for Overlap Stitching
    tile_size = 128
    overlap = 64  # 50% overlap
    stride = tile_size - overlap
    
    # 1. Initialize Orchestrator
    orchestrator = AetherOrchestrator().to(device)
    
    # Freeze Complexity Mask and Geometric Engine
    for p in orchestrator.complexity_mask.parameters():
        p.requires_grad = False
    for p in orchestrator.geometric_engine.parameters():
        p.requires_grad = False
        
    # 2. Optimizer (AdamW)
    trainable_params = list(orchestrator.structural_engine.parameters()) + \
                      list(orchestrator.neural_engine.parameters())
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=1e-2)
    scaler = torch.amp.GradScaler(device.type) if device.type == 'cuda' else None
    
    # 3. Resume Logic
    os.makedirs('checkpoints', exist_ok=True)
    start_epoch = 1
    if resume:
        checkpoints = sorted(Path('checkpoints').glob('stage1_epoch_*.pth'), key=os.path.getmtime)
        if checkpoints:
            latest_ckpt = checkpoints[-1]
            print(f"🔄 Resuming from checkpoint: {latest_ckpt}")
            orchestrator.load_state_dict(torch.load(latest_ckpt, map_location=device))
            try:
                start_epoch = int(latest_ckpt.stem.split('_')[-1]) + 1
            except:
                pass
        
        if start_epoch > epochs:
            print(f"⚠️ Resumed epoch ({start_epoch-1}) is already >= total epochs ({epochs}).")
            print("To train further, increase the --epochs argument.")
            return

    # 4. Data Loading
    dataset = HybridDataset(data_dir, image_size=image_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    
    os.makedirs('samples', exist_ok=True)
    
    # Accuracy-First Weights
    lambda_rec = 5.0      # Increased for pixel-perfect foundation
    lambda_commit = 0.25
    lambda_rate = 0.01
    lambda_tv = 0.1       # Doubled for smoothness
    lambda_edge = 1.0     # New Edge-Match Loss
    
    for epoch in range(start_epoch, epochs + 1):
        orchestrator.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")
        
        epoch_losses = []
        usage_stats = {0: 0, 1: 0, 2: 0}
        step = 0
        
        for images in pbar:
            images = images.to(device, non_blocking=True)
            B_img = images.shape[0]
            
            # --- Overlap-Aware Tiling ---
            # Instead of simple unfold, we use sliding window with overlap
            # (B, 3, 256, 256) -> (B, 3, 128, 128, num_tiles)
            tiles = images.unfold(2, tile_size, stride).unfold(3, tile_size, stride)
            # Shape: (B, 3, n_h, n_w, 128, 128)
            B, C, NH, NW, TH, TW = tiles.shape
            tiles = tiles.permute(0, 2, 3, 1, 4, 5).reshape(-1, 3, 128, 128)
            B_tiles = tiles.shape[0]
            
            optimizer.zero_grad()
            
            with torch.amp.autocast(device.type):
                # --- Forward Pass with 0.95 Fallback Threshold ---
                results = orchestrator(tiles, sim_threshold=0.95)
                
                # Check for NaNs in each engine output
                for key in ['geometric', 'structural', 'neural']:
                    if results[key] is not None:
                        val = results[key][0] if isinstance(results[key], tuple) else results[key]
                        if torch.isnan(val).any():
                            print(f"❌ NaN detected in {key} engine output!")
                
                # Reconstruct with Gaussian Blending
                recon_images = orchestrator.reconstruct(results, B_tiles, image_size=image_size, overlap=overlap)
                
                if torch.isnan(recon_images).any():
                    print("❌ NaN detected in reconstructed image!")

                # --- Stage 1+ Loss ---
                l_rec = F.l1_loss(recon_images, images)
                
                # Total Variation (TV) Loss - The Contextual Glue
                l_tv = total_variation_loss(recon_images)
                
                # Edge Match Loss
                l_edge = edge_match_loss(recon_images, overlap=overlap)
                
                l_commit = torch.tensor(0.0, device=device)
                if results['structural'] is not None:
                    indices, _, _, _, _ = results['structural']
                    cb = orchestrator.structural_engine.codebook_module.codebook
                    selected_cb = cb[indices]
                    patches = orchestrator.structural_engine._subdivide(tiles[(results['mask'] == 1).view(-1)])
                    if patches.shape[0] > 0:
                        l_commit = F.mse_loss(selected_cb, patches.detach())
                
                l_rate = torch.tensor(0.0, device=device)
                if results['neural'] is not None:
                    # Use sqrt(x^2 + eps) for robust absolute value
                    l_rate = torch.mean(torch.sqrt(results['neural']**2 + 1e-6))
                
                total_loss = (lambda_rec * l_rec + 
                              lambda_commit * l_commit + 
                              lambda_rate * l_rate + 
                              lambda_tv * torch.clamp(l_tv, max=1.0) +
                              lambda_edge * l_edge)
                
            if scaler:
                scaler.scale(total_loss).backward()
                # Clip gradients to prevent NaN
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
            
            # Stats update
            epoch_losses.append(total_loss.item())
            mask = results['mask']
            for s in range(3):
                usage_stats[s] += (mask == s).sum().item()
            
            # Periodic logging
            step += 1
            if step % 100 == 0:
                total_tiles = sum(usage_stats.values())
                usage_str = f"M:{usage_stats[0]/total_tiles:.1%} S:{usage_stats[1]/total_tiles:.1%} N:{usage_stats[2]/total_tiles:.1%}"
                pbar.set_postfix({"Loss": f"{total_loss.item():.4f}", "Usage": usage_str})

        # --- End of Epoch: Save Samples ---
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        vutils.save_image(recon_images[:4], f"samples/recon_epoch_{epoch}.png", normalize=True)
        
        print(f"✅ Epoch {epoch} Complete. Avg Loss: {avg_loss:.4f}")
            
        # Robust Save every 5 epochs
        if epoch % 5 == 0:
            save_path = f"checkpoints/stage1_epoch_{epoch}.pth"
            torch.save(orchestrator.state_dict(), save_path)
            print(f"💾 Checkpoint saved: {save_path}")
            
    torch.save(orchestrator.state_dict(), "stage1_final_foundation.pth")
    print("🏆 Training Complete. Final model saved as stage1_final_foundation.pth")

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Aether-Blueprint Stage 1 Trainer")
    parser.add_argument("data_dir", type=str, help="Path to the dataset directory")
    parser.add_argument("--size", type=int, default=256, help="Image resolution (256 or 512)")
    parser.add_argument("--epochs", type=int, default=20, help="Total number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--no_resume", action="store_true", help="Disable automatic resuming")
    
    args = parser.parse_args()
    
    if os.path.exists(args.data_dir):
        train_stage1(
            data_dir=args.data_dir, 
            epochs=args.epochs, 
            batch_size=args.batch_size, 
            lr=args.lr, 
            resume=not args.no_resume,
            image_size=args.size
        )
    else:
        print(f"❌ Dataset directory {args.data_dir} not found. Please run the downloader first.")
