# 🚀 Aether-Blueprint v3.0: Google Colab Deployment Guide

This guide will help you run the **Aether-Blueprint v3.0** training and inference pipeline on Google Colab to leverage the **Tesla T4 GPU**.

## 1. Environment Setup
Open a new notebook in Google Colab and set the runtime to **GPU** (`Runtime` -> `Change runtime type` -> `T4 GPU`).

### Clone the Repository
Run the following to pull the official engine:
```bash
!git clone https://github.com/ethcocoder/paragram-engin.git
%cd paragram-engin
```

### Install Dependencies
Run the following to install the lightweight mobile requirements:
```python
!pip install torch torchvision numpy pillow pyyaml onnx onnxruntime tqdm scipy lpips
```

## 2. Training Stage 1+: Stability & Hybrid Logic

Stage 1+ trains the **Structural Dictionary** (Pattern Memory) and the **Neural Engine** (Mobile Specialist) while keeping the Math Wizard and Complexity Mask frozen.

### Key Improvements (v3.1)
- **Threshold Curriculum**: Similarity threshold ramps from `0.70 → 0.90` over 10 epochs, allowing the Structural Engine to contribute early.
- **Loss Softening**: `λ_rec=2.0`, `λ_edge=0.1` — gentler training prevents NaN artifacts.
- **Gradient Clipping**: Set to `0.5` to prevent exploding values at tile boundaries.
- **P2P Simulation**: At every epoch end, the system packs a test image into `.padox` binary and prints the compression ratio in KB.

### Prepare the Full Flickr8k Dataset
To train the foundation on 8,000+ real images (~1.1GB), run:
```bash
!python dataset_downloader.py --flickr8k
```

### Run the Stable Trainer (T4 Optimized)
```bash
# Fresh start (recommended for first run):
!rm -rf checkpoints samples stage1_final_foundation.pth
!python src/train/trainer_stage1.py dataset/flickr8k/images --size 256 --epochs 30 --batch_size 8 --no_resume
```

**To resume training:**
```bash
!python src/train/trainer_stage1.py dataset/flickr8k/images --size 256 --epochs 30 --batch_size 8
```

**To train at full 512 resolution (slower, for Stage 2 prep):**
```bash
!python src/train/trainer_stage1.py dataset/flickr8k/images --size 512 --epochs 10 --batch_size 4
```

## 3. What to Expect in the Logs

Each epoch will print:
```
✅ Epoch 5 | Loss: 0.1234 | τ=0.80 | M:12.3% S:45.6% N:42.1%
  📦 P2P Simulation  |  .padox: 23.4 KB  |  Raw: 192 KB  |  Ratio: 8.2×  |  Engines: M:15% S:50% N:35%
```
- **τ** = current curriculum threshold (ramps from 0.70 to 0.90)
- **M/S/N** = Math / Structural / Neural engine usage %
- **Ratio** = compression ratio (higher = better)

### Goals
- **S (Structural) usage should rise above 30%** as the codebook learns real textures.
- **No black artifacts** — the epsilon hardening and gradient clipping prevent NaN black holes.
- **Compression ratio should improve** each epoch as the codebook fills with meaningful patterns.

## 4. Current System Status

| Component | File | Status |
|-----------|------|--------|
| **Complexity Mask** | `src/core/complexity_mask.py` | ✅ **Functional** (Fourier-Variance) |
| **Math Wizard** | `src/engines/geometric/surface_fit.py` | ✅ **Functional** (Polynomial) |
| **Pattern Memory** | `src/engines/structural/dictionary.py` | ✅ **Functional** (Codebook) |
| **Mobile Specialist** | `src/engines/neural/lightweight.py` | ✅ **Functional** (GhostNet) |
| **Hybrid Brain** | `src/core/orchestrator.py` | ✅ **Functional** (Curriculum Routing) |
| **Training Pipeline** | `src/train/trainer_stage1.py` | ✅ **Functional** (Stage 1+) |
| **Binary Packing** | `src/core/blueprint_format.py` | ✅ **Functional** (.padox P2P) |
| **Mobile Bridge** | `deployment/` | 🚧 *Docstrings only* |

## 5. Checking Results
- **Reconstruction samples**: `samples/recon_epoch_{N}.png`
- **P2P decoded images**: `samples/p2p_epoch_{N}.png`
- **Packed binary files**: `samples/p2p_epoch_{N}.padox`
- **Checkpoints**: `checkpoints/stage1_epoch_{N}.pth` (saved every 5 epochs)

## 6. Verification
After the first epoch, check:
1. `samples/recon_epoch_1.png` — visual quality of the hybrid reconstruction.
2. `samples/p2p_epoch_1.padox` — the actual compressed binary file.
3. The log line showing compression ratio (e.g., `Ratio: 5.2×`).

---
**Tip:** If you are using Google Drive to store checkpoints, mount it first:
```python
from google.colab import drive
drive.mount('/content/drive')
```
And update the save path in `trainer_stage1.py`.
