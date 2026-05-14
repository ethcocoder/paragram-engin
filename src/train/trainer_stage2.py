"""
trainer_stage2.py — Aether-Blueprint v3.0
==========================================
Stage 2: Perceptual Refinement Trainer.

Responsibility:
    Fine-tune the model for VISUAL FIDELITY after the structural foundation
    is established in Stage 1.

    Key differences from Stage 1:
        - Loads `stage1_final_foundation.pth` as starting point.
        - Uses PerceptualLoss (LPIPS + MS-SSIM + L1) instead of raw L1.
        - Unfreezes SwinWindowAttention and TextureCodebook for refinement.
        - Low learning rate (2e-5) to preserve the foundation.
        - Focused edge_match_loss on Gaussian overlap zones.
        - 10 epochs of perceptual polish.

    The result: reconstructions that are visually indistinguishable from
    the source, not just numerically close.
"""

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
import math

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.core.orchestrator import HybridOrchestrator
from src.core.complexity_mask import ComplexityMask
from src.engines.geometric.surface_fit import GeometricEngine
from src.engines.structural.dictionary import StructuralEngine
from src.engines.neural.lightweight import LightweightNeuralEngine
from src.utils.perceptual import PerceptualLoss
from src.core.blueprint_format import pack, save, compression_ratio


# ──────────────────────────────────────────────────────────────────────────────
# Dataset (reused from Stage 1)
# ──────────────────────────────────────────────────────────────────────────────

class Stage2Dataset(Dataset):
    """
    Dataset for Stage 2 perceptual refinement.
    Uses lighter augmentation than Stage 1 to preserve detail.
    """
    def __init__(self, data_dir, image_size=512):
        self.data_dir = Path(data_dir)
        self.image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
            self.image_paths.extend(list(self.data_dir.glob(f'**/{ext}')))
        # Filter out OS artifacts
        self.image_paths = [
            p for p in self.image_paths
            if not p.name.startswith('.') and not p.name.startswith('__')
        ]
        print(f"  📂 Found {len(self.image_paths)} images for Stage 2.")

        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
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
# Loss: Gaussian-Weighted Edge Match (targeted at overlap zones)
# ──────────────────────────────────────────────────────────────────────────────

EPS = 1e-6


def gaussian_overlap_edge_loss(recon: torch.Tensor,
                                target: torch.Tensor,
                                tile_size: int = 128,
                                overlap: int = 64) -> torch.Tensor:
    """
    Compute edge-match loss ONLY in the Gaussian overlap zones.
    
    This ensures transitions between Math/Structural/Neural tiles are
    100% invisible by penalizing discontinuities specifically where
    tiles overlap and Gaussian blending occurs.

    The overlap zone is the strip of width `overlap` at each tile boundary.
    We weight the loss by a Gaussian falloff so errors near tile centers
    (where blending weight is near 1.0) are penalized less than errors
    at tile edges (where two tiles' Gaussian weights compete).

    Args:
        recon  : (B, C, H, W) reconstructed image.
        target : (B, C, H, W) ground truth image.
        tile_size : tile edge length.
        overlap   : overlap in pixels.

    Returns:
        Scalar loss focusing on overlap regions.
    """
    B, C, H, W = recon.shape
    stride = tile_size - overlap

    # Build a weight mask that is high in overlap zones, low elsewhere
    weight_h = torch.zeros(H, device=recon.device)
    weight_w = torch.zeros(W, device=recon.device)

    # Mark overlap regions along height
    positions_h = list(range(0, H - tile_size + 1, stride))
    for pos in positions_h:
        # The overlap zone is [pos + stride, pos + tile_size)
        # which is where this tile overlaps with the next tile
        oz_start = pos + stride
        oz_end = min(pos + tile_size, H)
        if oz_start < oz_end:
            sigma = overlap / 4.0
            zone_len = oz_end - oz_start
            zone_mid = zone_len / 2.0
            idx = torch.arange(zone_len, device=recon.device, dtype=torch.float32)
            gauss = torch.exp(-((idx - zone_mid) ** 2) / (2 * sigma ** 2))
            weight_h[oz_start:oz_end] = torch.maximum(
                weight_h[oz_start:oz_end], gauss
            )

    # Mark overlap regions along width
    positions_w = list(range(0, W - tile_size + 1, stride))
    for pos in positions_w:
        oz_start = pos + stride
        oz_end = min(pos + tile_size, W)
        if oz_start < oz_end:
            sigma = overlap / 4.0
            zone_len = oz_end - oz_start
            zone_mid = zone_len / 2.0
            idx = torch.arange(zone_len, device=recon.device, dtype=torch.float32)
            gauss = torch.exp(-((idx - zone_mid) ** 2) / (2 * sigma ** 2))
            weight_w[oz_start:oz_end] = torch.maximum(
                weight_w[oz_start:oz_end], gauss
            )

    # Combine h and w weights into a 2D mask (outer sum, not product,
    # so either direction's overlap zone gets attention)
    weight_mask = (weight_h.unsqueeze(1) + weight_w.unsqueeze(0)).clamp(max=1.0)
    weight_mask = weight_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

    # Weighted L1 difference in overlap zones
    diff = torch.sqrt((recon - target) ** 2 + EPS)
    weighted_diff = diff * weight_mask
    
    # Normalize by the weight sum to avoid scale dependency
    weight_sum = weight_mask.sum().clamp(min=1.0)
    return weighted_diff.sum() / (weight_sum * B * C)


# ──────────────────────────────────────────────────────────────────────────────
# P2P Verification (same as Stage 1 but with entropy coding)
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_p2p_verification(orchestrator: HybridOrchestrator,
                          test_image: torch.Tensor,
                          epoch: int,
                          device: torch.device) -> tuple:
    """
    Run Point-to-Point simulation: encode → pack → measure → decode → save.
    """
    orchestrator.eval()
    img = test_image.unsqueeze(0).to(device)
    B, C, H, W = img.shape

    results = orchestrator.encode(img)
    stats = results['stats']

    # Build payloads for packing
    geo_payload = {'coeffs': []}
    struct_payload = {'indices': np.array([], dtype=np.uint16),
                      'rotations': np.array([], dtype=np.uint8),
                      'gains': np.array([], dtype=np.float32),
                      'biases': np.array([], dtype=np.float32)}
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

    padox_bytes = pack(H, W, orchestrator.tile_size,
                       geo_payload, struct_payload, neural_payload)
    file_size_kb = len(padox_bytes) / 1024.0

    # Reconstruct
    recon = orchestrator.reconstruct(results, 1, (H, W))

    # Save outputs
    os.makedirs('samples_s2', exist_ok=True)
    save(f"samples_s2/s2_epoch_{epoch}.padox", padox_bytes)
    vutils.save_image(recon, f"samples_s2/s2_epoch_{epoch}.png", normalize=False)

    # Compute ratio
    raw_kb = (H * W * C) / 1024.0
    ratio = raw_kb / max(file_size_kb, 0.01)

    print(f"  📦 P2P  |  .padox: {file_size_kb:.1f} KB  |  "
          f"Raw: {raw_kb:.0f} KB  |  Ratio: {ratio:.1f}×  |  "
          f"G:{stats['pct_geometric']:.0f}% S:{stats['pct_structural']:.0f}% "
          f"N:{stats['pct_neural']:.0f}%")

    orchestrator.train()
    return file_size_kb, ratio


# Need numpy for payload building
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Main Training Loop
# ──────────────────────────────────────────────────────────────────────────────

def train_stage2(data_dir: str,
                 checkpoint_path: str = 'stage1_final_foundation.pth',
                 epochs: int = 10,
                 batch_size: int = 4,
                 lr: float = 2e-5,
                 image_size: int = 256):
    """
    Stage 2: Perceptual Refinement.

    Loads Stage 1 foundation and fine-tunes with perceptual loss
    (LPIPS + MS-SSIM + L1) and Gaussian overlap edge matching.

    Args:
        data_dir        : path to training images.
        checkpoint_path : Stage 1 checkpoint to load.
        epochs          : number of fine-tuning epochs (default 10).
        batch_size      : training batch size (default 4, lower for LPIPS memory).
        lr              : learning rate (default 2e-5).
        image_size      : training image resolution.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🎨 Stage 2 (Perceptual Refinement) on {device} | Size: {image_size}")

    tile_size = 128
    overlap   = 64
    stride    = tile_size - overlap

    # ── 1. Build Orchestrator ────────────────────────────────────────────────
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

    # ── 2. Load Stage 1 Checkpoint ──────────────────────────────────────────
    if os.path.exists(checkpoint_path):
        print(f"  🔄 Loading Stage 1 foundation: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
        # Try loading with flexible key matching
        try:
            orchestrator.load_state_dict(state_dict, strict=False)
        except Exception as e:
            print(f"  ⚠️ Partial load (expected for architecture changes): {e}")
            # Manual key mapping for backwards compat
            own_state = orchestrator.state_dict()
            for name, param in state_dict.items():
                if name in own_state and own_state[name].shape == param.shape:
                    own_state[name].copy_(param)
            print("  ✅ Loaded compatible weights.")
    else:
        print(f"  ⚠️ No Stage 1 checkpoint found at {checkpoint_path}. Training from scratch.")

    # ── 3. Freeze / Unfreeze Strategy ───────────────────────────────────────
    # FREEZE: Complexity Mask (routing is stable) + Geometric Engine (exact math)
    for p in orchestrator.mask_module.parameters():
        p.requires_grad = False
    for p in orchestrator.geo_engine.parameters():
        p.requires_grad = False

    # UNFREEZE: SwinWindowAttention (in neural engine) + TextureCodebook (in structural)
    # These are already unfrozen by default, but let's be explicit:
    for name, p in orchestrator.neural_engine.named_parameters():
        p.requires_grad = True  # All neural params trainable
    for name, p in orchestrator.struct_engine.named_parameters():
        p.requires_grad = True  # Codebook + matching params

    trainable_params = [p for p in orchestrator.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    print(f"  🧠 Trainable parameters: {n_trainable:,}")

    # ── 4. Optimizer ────────────────────────────────────────────────────────
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.1)
    scaler = torch.amp.GradScaler(device.type) if device.type == 'cuda' else None

    # ── 5. Loss Functions ───────────────────────────────────────────────────
    perceptual_criterion = PerceptualLoss(
        lpips_weight=0.5,
        ssim_weight=0.4,
        mse_weight=0.1,
    ).to(device)

    # Freeze LPIPS backbone (it's a pretrained evaluator, not trainable)
    for p in perceptual_criterion.parameters():
        p.requires_grad = False

    # Loss weights for Stage 2
    lambda_perceptual = 1.0       # Primary: perceptual quality
    lambda_edge       = 0.3       # Gaussian overlap stitching guard
    lambda_rate       = 0.005     # Gentle rate penalty (don't crush quality)

    # ── 6. Data Loading ─────────────────────────────────────────────────────
    dataset = Stage2Dataset(data_dir, image_size=image_size)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True
    )

    os.makedirs('samples_s2', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)

    # Pick a test image for P2P verification
    p2p_test_image = None

    # ── 7. Training Loop ────────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        orchestrator.train()

        pbar = tqdm(dataloader, desc=f"Stage2 Epoch {epoch}/{epochs}")
        epoch_losses = []
        epoch_perceptual = []
        epoch_edge = []
        step = 0

        for images in pbar:
            images = images.to(device, non_blocking=True)

            if p2p_test_image is None:
                p2p_test_image = images[0].detach().cpu()

            optimizer.zero_grad()

            with torch.amp.autocast(device.type):
                # Forward: encode → reconstruct
                results = orchestrator.encode(images)
                stats = results['stats']
                recon = orchestrator.reconstruct(results, images.shape[0], (images.shape[2], images.shape[3]))

                # ── NaN Sentinel ────────────────────────────────────────
                nan_detected = False
                for key in ['geo', 'struct', 'neural']:
                    if key in stats and stats.get(f'pct_{key}', 0) > 0:
                        # Check results dict if it exists in local scope or via orchestrator
                        # Actually, let's check recon directly first, but also check latents
                        pass

                if torch.isnan(recon).any():
                    print(f"  ❌ NaN in reconstruction — skipping batch.")
                    optimizer.zero_grad()
                    continue

                # ── Perceptual Loss ─────────────────────────────────────
                l_perceptual, loss_dict = perceptual_criterion(recon, images)

                # ── Gaussian Overlap Edge Loss ──────────────────────────
                l_edge = gaussian_overlap_edge_loss(
                    recon, images, tile_size=tile_size, overlap=overlap
                )

                # ── Rate Penalty (neural latents) ───────────────────────
                # Encourage compact latent representations
                l_rate = torch.tensor(0.0, device=device)
                if 'neural' in results and results['neural'] is not None:
                    latents = results['neural']
                    if torch.isnan(latents).any():
                        print("  ❌ NaN in neural latents! Skipping.")
                        optimizer.zero_grad()
                        continue
                    l_rate = torch.mean(torch.sqrt(latents ** 2 + EPS))

                # ── Total Loss ──────────────────────────────────────────
                total_loss = (lambda_perceptual * l_perceptual
                            + lambda_edge       * torch.clamp(l_edge, max=2.0)
                            + lambda_rate       * l_rate)

                if torch.isnan(total_loss):
                    print(f"  ❌ NaN in total loss! (P:{l_perceptual.item():.4f}, E:{l_edge.item():.4f}, R:{l_rate.item():.4f})")
                    optimizer.zero_grad()
                    continue

            # ── Backward ────────────────────────────────────────────────
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

            # Stats
            epoch_losses.append(total_loss.item())
            epoch_perceptual.append(l_perceptual.item())
            epoch_edge.append(l_edge.item())

            step += 1
            if step % 20 == 0:
                pbar.set_postfix({
                    'Loss': f"{total_loss.item():.4f}",
                    'LPIPS': f"{loss_dict['lpips']:.3f}",
                    'SSIM': f"{loss_dict['ms_ssim']:.3f}",
                    'Edge': f"{l_edge.item():.4f}",
                })

        # ── End of Epoch ────────────────────────────────────────────────────
        scheduler.step()
        avg_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        avg_perc = sum(epoch_perceptual) / max(len(epoch_perceptual), 1)
        avg_edge = sum(epoch_edge) / max(len(epoch_edge), 1)
        current_lr = scheduler.get_last_lr()[0]

        print(f"  ✅ Epoch {epoch} | Loss: {avg_loss:.4f} | "
              f"Perceptual: {avg_perc:.4f} | Edge: {avg_edge:.4f} | "
              f"LR: {current_lr:.2e}")

        # Save reconstruction sample
        with torch.no_grad():
            vutils.save_image(
                recon[:4], f"samples_s2/recon_s2_epoch_{epoch}.png",
                normalize=False
            )

        # P2P Verification
        if p2p_test_image is not None:
            try:
                run_p2p_verification(orchestrator, p2p_test_image, epoch, device)
            except Exception as e:
                print(f"  ⚠️ P2P verification failed: {e}")

        # Checkpoint every 2 epochs
        if epoch % 2 == 0 or epoch == epochs:
            ckpt_path = f"checkpoints/stage2_epoch_{epoch}.pth"
            torch.save(orchestrator.state_dict(), ckpt_path)
            print(f"  💾 Checkpoint saved: {ckpt_path}")

    # ── Final Save ──────────────────────────────────────────────────────────
    final_path = "stage2_perceptual.pth"
    torch.save(orchestrator.state_dict(), final_path)
    print(f"🏆 Stage 2 Complete. Final model saved as {final_path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Aether-Blueprint Stage 2: Perceptual Refinement")
    parser.add_argument("data_dir", type=str, help="Path to the dataset directory")
    parser.add_argument("--checkpoint", type=str, default="stage1_final_foundation.pth",
                        help="Path to Stage 1 checkpoint")
    parser.add_argument("--size", type=int, default=256, help="Image resolution")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")

    args = parser.parse_args()

    if os.path.exists(args.data_dir):
        train_stage2(
            data_dir=args.data_dir,
            checkpoint_path=args.checkpoint,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            image_size=args.size,
        )
    else:
        print(f"❌ Dataset directory {args.data_dir} not found.")
