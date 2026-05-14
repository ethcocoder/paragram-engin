"""
trainer_stage1.py — Aether-Blueprint v3.0
==========================================
Stage 1: Foundation Training Trainer.

Responsibility:
    Train the Structural and Neural engines from scratch to establish a solid
    hybrid compression foundation. 

    Key objectives:
        - Reconstruct tiles accurately (L1 Loss).
        - Enforce spatial smoothness and edge continuity (TV + Edge-Match Loss).
        - Optimize the Structural Engine's codebook (Commitment Loss).
        - Penalize large neural latent magnitudes (Rate Penalty).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torchvision.utils as vutils

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.core.orchestrator import HybridOrchestrator
from src.core.complexity_mask import ComplexityMask
from src.engines.geometric.surface_fit import GeometricEngine
from src.engines.structural.dictionary import StructuralEngine
from src.engines.neural.lightweight import LightweightNeuralEngine
from src.core.blueprint_format import pack, save, compression_ratio

# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class HybridDataset(Dataset):
    """
    Robust dataset for Aether-Blueprint training.
    """
    def __init__(self, data_dir, image_size=512):
        self.data_dir = Path(data_dir)
        self.image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
            self.image_paths.extend(list(self.data_dir.glob(f'**/{ext}')))
        self.image_paths = [
            p for p in self.image_paths
            if not p.name.startswith('.') and not p.name.startswith('__')
        ]
        
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
        except Exception:
            return torch.zeros(3, 512, 512)

# ──────────────────────────────────────────────────────────────────────────────
# Loss Functions with Epsilon Hardening
# ──────────────────────────────────────────────────────────────────────────────

EPS = 1e-6

def total_variation_loss(img):
    tv_h = torch.sqrt(torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2) + EPS).mean()
    tv_w = torch.sqrt(torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2) + EPS).mean()
    return tv_h + tv_w

def edge_match_loss(img, overlap=16):
    h_diff = img[:, :, :, 1:] - img[:, :, :, :-1]
    v_diff = img[:, :, 1:, :] - img[:, :, :-1, :]
    return (torch.mean(torch.sqrt(h_diff ** 2 + EPS))
          + torch.mean(torch.sqrt(v_diff ** 2 + EPS)))

# ──────────────────────────────────────────────────────────────────────────────
# P2P Simulation
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_p2p_simulation(orchestrator, test_image, epoch, device):
    orchestrator.eval()
    img = test_image.unsqueeze(0).to(device)
    B, C, H, W = img.shape

    results = orchestrator.encode(img)
    stats = results['stats']

    geo_payload = {'coeffs': []}
    struct_payload = {
        'indices': np.array([], dtype=np.uint16),
        'rotations': np.array([], dtype=np.uint8),
        'gains': np.array([], dtype=np.float32),
        'biases': np.array([], dtype=np.float32)
    }
    neural_payload = {'latents': np.zeros((0, 128), dtype=np.float32)}

    if 'geo' in results and results['geo'] is not None:
        coeffs = results['geo'].detach().cpu().numpy()
        geo_payload['coeffs'] = [coeffs[i] for i in range(coeffs.shape[0])]

    if 'struct' in results and results['struct'] is not None:
        s = results['struct']
        struct_payload = {
            'indices':   s['indices'].detach().cpu().numpy().astype(np.uint16),
            'rotations': s['rotations'].detach().cpu().numpy().astype(np.uint8),
            'gains':     s['gains'].detach().cpu().numpy().astype(np.float32),
            'biases':    s['biases'].detach().cpu().numpy().astype(np.float32),
        }

    if 'neural' in results and results['neural'] is not None:
        neural_payload['latents'] = results['neural'].detach().cpu().numpy()

    padox_bytes = pack(H, W, orchestrator.tile_size, geo_payload, struct_payload, neural_payload)
    file_size_kb = len(padox_bytes) / 1024.0

    os.makedirs('samples', exist_ok=True)
    save(f"samples/p2p_epoch_{epoch}.padox", padox_bytes)

    recon = orchestrator.reconstruct(results, 1, (H, W))
    vutils.save_image(recon, f"samples/p2p_epoch_{epoch}.png", normalize=False)

    raw_size_kb = (H * W * C) / 1024.0
    ratio = raw_size_kb / max(file_size_kb, 0.01)

    print(f"  📦 P2P Simulation  |  .padox: {file_size_kb:.1f} KB  |  "
          f"Raw: {raw_size_kb:.0f} KB  |  Ratio: {ratio:.1f}×  |  "
          f"G:{stats['pct_geometric']:.0f}% S:{stats['pct_structural']:.0f}% N:{stats['pct_neural']:.0f}%")

    orchestrator.train()
    return file_size_kb, ratio

# ──────────────────────────────────────────────────────────────────────────────
# Main Training Loop
# ──────────────────────────────────────────────────────────────────────────────

def train_stage1(data_dir, epochs=20, batch_size=8, lr=1e-4, resume=True, image_size=256):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Stage 1+ (Stability & Hybrid Logic) on {device} | Size: {image_size}")
    
    tile_size = 128
    overlap = 64
    
    geo_engine    = GeometricEngine(tile_size=tile_size)
    struct_engine = StructuralEngine(codebook_size=512, patch_size=32, tile_size=tile_size)
    neural_engine = LightweightNeuralEngine(tile_size=tile_size, latent_dim=128)

    orchestrator = HybridOrchestrator(
        geo_engine=geo_engine,
        struct_engine=struct_engine,
        neural_engine=neural_engine,
        tile_size=tile_size,
        overlap=0.5,
    ).to(device)
    
    for p in orchestrator.mask_module.parameters():
        p.requires_grad = False
    for p in orchestrator.geo_engine.parameters():
        p.requires_grad = False
        
    trainable_params = list(orchestrator.struct_engine.parameters()) + \
                      list(orchestrator.neural_engine.parameters())
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=1e-2)
    scaler = torch.amp.GradScaler(device.type) if device.type == 'cuda' else None
    
    os.makedirs('checkpoints', exist_ok=True)
    start_epoch = 1
    if resume:
        checkpoints = sorted(Path('checkpoints').glob('stage1_epoch_*.pth'), key=os.path.getmtime)
        if checkpoints:
            latest_ckpt = checkpoints[-1]
            print(f"🔄 Resuming from checkpoint: {latest_ckpt}")
            state_dict = torch.load(latest_ckpt, map_location=device, weights_only=False)
            try:
                orchestrator.load_state_dict(state_dict, strict=False)
            except Exception as e:
                print(f"  ⚠️ Partial load: {e}")
                own_state = orchestrator.state_dict()
                for name, param in state_dict.items():
                    if name in own_state and own_state[name].shape == param.shape:
                        own_state[name].copy_(param)
            try:
                start_epoch = int(latest_ckpt.stem.split('_')[-1]) + 1
            except:
                pass

    dataset = HybridDataset(data_dir, image_size=image_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    
    os.makedirs('samples', exist_ok=True)
    
    lambda_rec    = 2.0
    lambda_commit = 0.25
    lambda_rate   = 0.01
    lambda_tv     = 0.05
    lambda_edge   = 0.1
    
    p2p_test_image = None

    for epoch in range(start_epoch, epochs + 1):
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        orchestrator.train()
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")
        epoch_losses = []
        step = 0
        
        for images in pbar:
            images = images.to(device, non_blocking=True)
            if p2p_test_image is None:
                p2p_test_image = images[0].detach().cpu()
            
            optimizer.zero_grad()
            
            with torch.amp.autocast(device.type):
                results = orchestrator.encode(images)
                
                nan_detected = False
                for key in ['geo', 'struct', 'neural']:
                    if key in results and results[key] is not None:
                        val = results[key]
                        if isinstance(val, dict):
                            for k, v in val.items():
                                if isinstance(v, torch.Tensor) and torch.isnan(v).any():
                                    nan_detected = True
                                    break
                        elif isinstance(val, torch.Tensor) and torch.isnan(val).any():
                            nan_detected = True
                        if nan_detected:
                            print(f"  ❌ NaN detected in {key} engine output! Skipping batch.")
                            break
                
                if nan_detected:
                    optimizer.zero_grad()
                    continue
                
                B_img = images.shape[0]
                recon_images = orchestrator.reconstruct(results, B_img, image_size=(images.shape[2], images.shape[3]))
                
                if torch.isnan(recon_images).any():
                    print("  ❌ NaN in reconstruction — skipping batch.")
                    optimizer.zero_grad()
                    continue

                l_rec = F.l1_loss(recon_images, images)
                l_tv = total_variation_loss(recon_images)
                l_edge = edge_match_loss(recon_images, overlap=overlap)
                
                l_commit = torch.tensor(0.0, device=device)
                if 'struct' in results and results['struct'] is not None:
                    indices = results['struct']['indices']
                    cb = orchestrator.struct_engine.codebook.weight
                    selected_cb = cb[indices]
                    
                    mask = results['routing_mask']
                    struct_mask = (mask == 1)
                    
                    tiles, _, _ = orchestrator.extract_tiles(images)
                    struct_tiles = tiles[struct_mask]
                    
                    if struct_tiles.shape[0] > 0:
                        patches = orchestrator.struct_engine._extract_patches(struct_tiles).view(struct_tiles.shape[0] * orchestrator.struct_engine.patches_per_tile, -1).float()
                        patches = F.normalize(patches, dim=1)
                        selected_cb = F.normalize(selected_cb, dim=1)
                        
                        min_count = min(selected_cb.shape[0], patches.shape[0])
                        if min_count > 0:
                            l_commit = F.mse_loss(selected_cb[:min_count], patches[:min_count].detach())
                
                l_rate = torch.tensor(0.0, device=device)
                if 'neural' in results and results['neural'] is not None:
                    l_rate = torch.mean(torch.sqrt(results['neural'] ** 2 + EPS))
                
                total_loss = (lambda_rec * l_rec
                            + lambda_commit * l_commit
                            + lambda_rate * l_rate
                            + lambda_tv * torch.clamp(l_tv, max=2.0)
                            + lambda_edge * torch.clamp(l_edge, max=2.0))
                
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
            
            epoch_losses.append(total_loss.item())
            
            step += 1
            if step % 50 == 0:
                stats = results['stats']
                usage_str = f"G:{stats['pct_geometric']:.0f}% S:{stats['pct_structural']:.0f}% N:{stats['pct_neural']:.0f}%"
                pbar.set_postfix({"Loss": f"{total_loss.item():.4f}", "Usage": usage_str})

        avg_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        print(f"✅ Epoch {epoch} | Loss: {avg_loss:.4f}")

        vutils.save_image(recon_images[:4], f"samples/recon_epoch_{epoch}.png", normalize=False)

        if p2p_test_image is not None:
            try:
                run_p2p_simulation(orchestrator, p2p_test_image, epoch, device)
            except Exception as e:
                print(f"  ⚠️ P2P simulation failed: {e}")

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
    parser.add_argument("--size", type=int, default=256, help="Image resolution")
    parser.add_argument("--epochs", type=int, default=20, help="Total epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--no_resume", action="store_true", help="Disable auto-resume")
    args = parser.parse_args()
    
    if os.path.exists(args.data_dir):
        train_stage1(args.data_dir, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, resume=not args.no_resume, image_size=args.size)
    else:
        print(f"❌ Dataset directory {args.data_dir} not found.")
