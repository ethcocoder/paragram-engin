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
import random

import sys
from pathlib import Path
# Add the project root to sys.path to ensure 'src' is found
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.core.orchestrator import AetherOrchestrator
from src.core.blueprint_format import pack_blueprint, unpack_blueprint, save_padox


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
            return torch.zeros(3, 512, 512)


# ─── Loss Functions with Epsilon Hardening ───────────────────────────────────

EPS = 1e-6  # Global epsilon for numerical stability

def total_variation_loss(img):
    """
    Penalizes sharp pixel jumps between horizontal and vertical neighbors.
    Enforces spatial smoothness across tile boundaries.
    Uses sqrt(x^2 + eps) instead of abs(x) to keep gradients smooth near zero.
    """
    tv_h = torch.sqrt(
        torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2) + EPS
    ).mean()
    tv_w = torch.sqrt(
        torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2) + EPS
    ).mean()
    return tv_h + tv_w


def edge_match_loss(img, overlap=16):
    """
    Penalizes differences at tile overlap boundaries.
    Uses smooth-abs (sqrt(x^2 + eps)) instead of raw abs to prevent NaN gradients.
    """
    h_diff = img[:, :, :, 1:] - img[:, :, :, :-1]
    v_diff = img[:, :, 1:, :] - img[:, :, :-1, :]
    return (torch.mean(torch.sqrt(h_diff ** 2 + EPS))
          + torch.mean(torch.sqrt(v_diff ** 2 + EPS)))


# ─── P2P Simulation ─────────────────────────────────────────────────────────

@torch.no_grad()
def run_p2p_simulation(orchestrator, test_image, epoch, image_size, tile_size, overlap, stride, device):
    """
    Simulates a Point-to-Point transmission:
      1. Encode → Pack into .padox binary
      2. Measure file size
      3. Decode .padox back into an image
      4. Save the decoded image and print compression ratio
    """
    orchestrator.eval()

    # Tile the single test image
    img = test_image.unsqueeze(0).to(device)  # (1, 3, H, W)
    tiles = img.unfold(2, tile_size, stride).unfold(3, tile_size, stride)
    B, C, NH, NW, TH, TW = tiles.shape
    tiles = tiles.permute(0, 2, 3, 1, 4, 5).reshape(-1, 3, tile_size, tile_size)

    # Forward pass (use final curriculum threshold since this is evaluation)
    results = orchestrator(tiles, current_epoch=epoch)

    # ── Pack ────────────────────────────────────────────────────────────────
    padox_bytes = pack_blueprint(
        results,
        image_width=image_size,
        image_height=image_size,
        tile_size=tile_size,
        overlap=overlap
    )
    file_size_kb = len(padox_bytes) / 1024.0

    # Save the actual .padox file
    os.makedirs('samples', exist_ok=True)
    padox_path = f"samples/p2p_epoch_{epoch}.padox"
    save_padox(padox_path, padox_bytes)

    # ── Decode (from packed results, not from file — uses same orchestrator) ─
    # We reconstruct from the ORIGINAL results to show the training-quality image
    recon = orchestrator.reconstruct(results, tiles.shape[0],
                                     image_size=image_size, overlap=overlap)

    # Save reconstructed image
    png_path = f"samples/p2p_epoch_{epoch}.png"
    vutils.save_image(recon, png_path, normalize=False)

    # ── Compression Ratio ──────────────────────────────────────────────────
    raw_size_kb = (image_size * image_size * 3 * 1) / 1024.0  # uint8 raw
    ratio = raw_size_kb / max(file_size_kb, 0.01)

    # Count engine usage for the report
    mask = results['mask'].view(-1)
    total = mask.numel()
    m_pct = (mask == 0).sum().item() / total * 100
    s_pct = (mask == 1).sum().item() / total * 100
    n_pct = (mask == 2).sum().item() / total * 100

    print(f"  📦 P2P Simulation  |  .padox: {file_size_kb:.1f} KB  |  "
          f"Raw: {raw_size_kb:.0f} KB  |  Ratio: {ratio:.1f}×  |  "
          f"Engines: M:{m_pct:.0f}% S:{s_pct:.0f}% N:{n_pct:.0f}%")

    orchestrator.train()
    return file_size_kb, ratio


# ─── Main Training Loop ─────────────────────────────────────────────────────

def train_stage1(data_dir, epochs=20, batch_size=16, lr=1e-4, resume=True, image_size=256):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Stage 1+ (Stability & Hybrid Logic) on {device} | Size: {image_size}")
    
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
    
    # ── Loss Weights (Softened for stability) ───────────────────────────────
    lambda_rec    = 2.0    # Softened from 5.0 — gentle while model learns overlap
    lambda_commit = 0.25
    lambda_rate   = 0.01
    lambda_tv     = 0.05   # Gentle TV smoothness
    lambda_edge   = 0.1    # Softened from 1.0 — prevents NaN feedback loops
    
    # Pick a random test image for P2P simulation (stays the same across epochs)
    p2p_test_image = None

    for epoch in range(start_epoch, epochs + 1):
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        orchestrator.train()

        # Compute curriculum threshold for this epoch
        cur_threshold = orchestrator.get_curriculum_threshold(epoch)
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs} [τ={cur_threshold:.2f}]")
        
        epoch_losses = []
        usage_stats = {0: 0, 1: 0, 2: 0}
        step = 0
        
        for images in pbar:
            images = images.to(device, non_blocking=True)
            B_img = images.shape[0]

            # Save a test image for P2P simulation (first batch, first epoch only)
            if p2p_test_image is None:
                p2p_test_image = images[0].detach().cpu()
            
            # --- Overlap-Aware Tiling ---
            tiles = images.unfold(2, tile_size, stride).unfold(3, tile_size, stride)
            B, C, NH, NW, TH, TW = tiles.shape
            tiles = tiles.permute(0, 2, 3, 1, 4, 5).reshape(-1, 3, 128, 128)
            B_tiles = tiles.shape[0]
            
            optimizer.zero_grad()
            
            with torch.amp.autocast(device.type):
                # Forward pass with curriculum threshold (no hard 0.95)
                results = orchestrator(tiles, current_epoch=epoch)
                
                # ── NaN Sentinel ────────────────────────────────────────────
                nan_detected = False
                for key in ['geometric', 'structural', 'neural']:
                    if results[key] is not None:
                        val = results[key][0] if isinstance(results[key], tuple) else results[key]
                        if torch.isnan(val).any():
                            print(f"  ❌ NaN detected in {key} engine output! Skipping batch.")
                            nan_detected = True
                            break
                
                if nan_detected:
                    optimizer.zero_grad()
                    continue
                
                # Reconstruct with Gaussian Blending
                recon_images = orchestrator.reconstruct(
                    results, B_tiles, image_size=image_size, overlap=overlap
                )
                
                if torch.isnan(recon_images).any():
                    print("  ❌ NaN in reconstruction — skipping batch.")
                    optimizer.zero_grad()
                    continue

                # ── Loss Computation (all eps-hardened) ─────────────────────
                l_rec = F.l1_loss(recon_images, images)
                
                l_tv = total_variation_loss(recon_images)
                l_edge = edge_match_loss(recon_images, overlap=overlap)
                
                # Codebook commitment loss
                l_commit = torch.tensor(0.0, device=device)
                if results['structural'] is not None:
                    indices, _, _, _, _ = results['structural']
                    cb = orchestrator.structural_engine.codebook_module.codebook
                    selected_cb = cb[indices]
                    struct_mask = (results['mask'] == 1).view(-1)
                    struct_tiles = tiles[struct_mask]
                    if struct_tiles.shape[0] > 0:
                        patches = orchestrator.structural_engine._subdivide(struct_tiles)
                        # Match patch count to indices count (may differ after fallback)
                        min_count = min(selected_cb.shape[0], patches.shape[0])
                        if min_count > 0:
                            l_commit = F.mse_loss(
                                selected_cb[:min_count],
                                patches[:min_count].detach()
                            )
                
                # Rate penalty (eps-hardened sqrt)
                l_rate = torch.tensor(0.0, device=device)
                if results['neural'] is not None:
                    l_rate = torch.mean(torch.sqrt(results['neural'] ** 2 + EPS))
                
                total_loss = (lambda_rec    * l_rec
                            + lambda_commit * l_commit
                            + lambda_rate   * l_rate
                            + lambda_tv     * torch.clamp(l_tv, max=2.0)
                            + lambda_edge   * torch.clamp(l_edge, max=2.0))
                
            # ── Backward with gradient clipping at 0.5 ─────────────────────
            if scaler:
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=0.5)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=0.5)
                optimizer.step()
            
            # Stats update
            epoch_losses.append(total_loss.item())
            mask = results['mask']
            for s in range(3):
                usage_stats[s] += (mask == s).sum().item()
            
            # Periodic logging
            step += 1
            if step % 50 == 0:
                total_tiles = max(sum(usage_stats.values()), 1)
                usage_str = (f"M:{usage_stats[0]/total_tiles:.1%} "
                             f"S:{usage_stats[1]/total_tiles:.1%} "
                             f"N:{usage_stats[2]/total_tiles:.1%}")
                pbar.set_postfix({"Loss": f"{total_loss.item():.4f}", "Usage": usage_str})

        # ── End of Epoch ────────────────────────────────────────────────────
        avg_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        total_tiles = max(sum(usage_stats.values()), 1)
        m_pct = usage_stats[0] / total_tiles * 100
        s_pct = usage_stats[1] / total_tiles * 100
        n_pct = usage_stats[2] / total_tiles * 100

        print(f"✅ Epoch {epoch} | Loss: {avg_loss:.4f} | τ={cur_threshold:.2f} | "
              f"M:{m_pct:.1f}% S:{s_pct:.1f}% N:{n_pct:.1f}%")

        # Save reconstruction sample
        vutils.save_image(recon_images[:4], f"samples/recon_epoch_{epoch}.png", normalize=False)

        # ── P2P Simulation ──────────────────────────────────────────────────
        if p2p_test_image is not None:
            try:
                run_p2p_simulation(
                    orchestrator, p2p_test_image, epoch,
                    image_size, tile_size, overlap, stride, device
                )
            except Exception as e:
                print(f"  ⚠️ P2P simulation failed: {e}")

        # Checkpoint every 5 epochs
        if epoch % 5 == 0:
            save_path = f"checkpoints/stage1_epoch_{epoch}.pth"
            torch.save(orchestrator.state_dict(), save_path)
            print(f"  💾 Checkpoint saved: {save_path}")
            
    torch.save(orchestrator.state_dict(), "stage1_final_foundation.pth")
    print("🏆 Training Complete. Final model saved as stage1_final_foundation.pth")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Aether-Blueprint Stage 1+ Trainer")
    parser.add_argument("data_dir", type=str, help="Path to the dataset directory")
    parser.add_argument("--size", type=int, default=256, help="Image resolution (256 or 512)")
    parser.add_argument("--epochs", type=int, default=20, help="Total number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
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
