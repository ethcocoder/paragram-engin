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

def train_stage1(data_dir, epochs=20, batch_size=16, lr=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Scaling Stage 1 Training on {device} (T4 Optimized)")
    
    # 1. Initialize Orchestrator
    orchestrator = AetherOrchestrator().to(device)
    
    # Freeze Complexity Mask and Geometric Engine
    for p in orchestrator.complexity_mask.parameters():
        p.requires_grad = False
    for p in orchestrator.geometric_engine.parameters():
        p.requires_grad = False
        
    # 2. Optimizer (AdamW for better weight decay handling)
    trainable_params = list(orchestrator.structural_engine.parameters()) + \
                      list(orchestrator.neural_engine.parameters())
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=1e-2)
    scaler = torch.amp.GradScaler(device.type) if device.type == 'cuda' else None
    
    # 3. Data Loading
    dataset = HybridDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('samples', exist_ok=True)
    
    lambda_rec = 1.0
    lambda_commit = 0.25
    lambda_rate = 0.01
    
    for epoch in range(1, epochs + 1):
        orchestrator.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")
        
        epoch_losses = []
        usage_stats = {0: 0, 1: 0, 2: 0}
        step = 0
        
        for images in pbar:
            images = images.to(device, non_blocking=True)
            B_img = images.shape[0]
            
            # Tile images (512x512 -> 16 tiles of 128x128 per image)
            tiles = images.unfold(2, 128, 128).unfold(3, 128, 128)
            tiles = tiles.permute(0, 2, 3, 1, 4, 5).reshape(-1, 3, 128, 128)
            B_tiles = tiles.shape[0]
            
            optimizer.zero_grad()
            
            with torch.amp.autocast(device.type):
                # Forward Pass
                results = orchestrator(tiles)
                reconstructed_tiles = orchestrator.reconstruct(results, B_tiles)
                
                # Reassemble image for L1 loss
                recon_images = reconstructed_tiles.view(B_img, 4, 4, 3, 128, 128)
                recon_images = recon_images.permute(0, 3, 1, 4, 2, 5).reshape(B_img, 3, 512, 512)
                
                # --- Loss Calculation ---
                l_rec = F.l1_loss(recon_images, images)
                
                l_commit = torch.tensor(0.0, device=device)
                if results['structural'] is not None:
                    indices, _, _, _, _ = results['structural']
                    cb = orchestrator.structural_engine.codebook_module.codebook
                    selected_cb = cb[indices]
                    # commitment loss: pull codebook towards data
                    # (Simplified: MSE between codebook and input patches)
                    patches = orchestrator.structural_engine._subdivide(tiles[(results['mask'] == 1).view(-1)])
                    if patches.shape[0] > 0:
                        l_commit = F.mse_loss(selected_cb, patches.detach())
                
                l_rate = torch.tensor(0.0, device=device)
                if results['neural'] is not None:
                    l_rate = torch.mean(torch.abs(results['neural']))
                
                total_loss = lambda_rec * l_rec + lambda_commit * l_commit + lambda_rate * l_rate
                
            if scaler:
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
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
            
        if epoch % 5 == 0:
            torch.save(orchestrator.state_dict(), f"checkpoints/stage1_epoch_{epoch}.pth")
            
    torch.save(orchestrator.state_dict(), "stage1_final_foundation.pth")
    print("🏆 Training Complete. Final model saved as stage1_final_foundation.pth")
            
    torch.save(orchestrator.state_dict(), "stage1_foundation.pth")
    print("Training Complete. Final model saved as stage1_foundation.pth")

if __name__ == "__main__":
    # Placeholder for data dir
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "dataset/flickr8k"
    if os.path.exists(data_dir):
        train_stage1(data_dir)
    else:
        print(f"Dataset directory {data_dir} not found. Skipping execution.")
