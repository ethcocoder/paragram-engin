# 🚀 Aether-Blueprint v3.0: Google Colab Deployment Guide

This guide will help you run the **Aether-Blueprint v3.0** training and inference pipeline on Google Colab to leverage the **Tesla T4 GPU**.

**Status**: 🏆 The Final Production Phase is Complete! All modules (Stage 1, 2, 3, Mobile Export, Master CLI, and Entropy Coder) are fully functional.

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

## 2. Stage 1: Foundation Training

Stage 1 trains the **Structural Dictionary** (Pattern Memory) and the **Neural Engine** (Mobile Specialist) while keeping the Geometric Engine and Complexity Mask frozen.

### Prepare the Dataset
To train the foundation on real images, run:
```bash
!python dataset_downloader.py --flickr8k
```

### Run the Stable Trainer (T4 Optimized)
```bash
# Fresh start (recommended for first run):
!rm -rf checkpoints samples stage1_final_foundation.pth
!python src/train/trainer_stage1.py dataset/flickr8k/images --size 256 --epochs 20 --batch_size 8 --no_resume
```

**To resume training:**
```bash
!python src/train/trainer_stage2.py dataset/flickr8k/images --size 256 --epochs 20 --batch_size 16
```

## 3. Stage 2: Perceptual Refinement

Stage 2 focuses on visual fidelity using the **Perceptual Loss** (LPIPS + MS-SSIM + L1) and fine-tunes the `SwinWindowAttention` and `TextureCodebook`.

```bash
!python src/train/trainer_stage2.py dataset/flickr8k/images --checkpoint stage1_final_foundation.pth --epochs 10 --batch_size 4 --lr 2e-5
```

This will output `stage2_perceptual.pth`, the visually refined model.

## 4. Stage 3: Quantization-Aware Training (QAT)

Stage 3 prepares the model for INT8 deployment by simulating quantization noise during the forward pass.

```bash
!python src/train/trainer_stage3.py dataset/flickr8k/images --checkpoint stage2_perceptual.pth --epochs 5 --batch_size 4
```

This will output `stage3_qat_final.pth`.

## 5. Visual Teleportation via Master CLI

Once you have a trained model (Stage 2 or 3), you can use the Master CLI to teleport (compress and decompress) images.

### Compress and Decompress a Photo:
```bash
!python blueprint_master.py teleport --input /content/my_photo.jpg --output teleported.padox --checkpoint stage2_perceptual.pth
```
*This command encodes the image to a `.padox` binary, prints a detailed compression report, and decodes it back to a PNG.*

### Inspect the `.padox` Binary:
```bash
!python blueprint_master.py inspect --input teleported.padox
```

### Export for Mobile Deployment (ONNX + INT8):
```bash
!python blueprint_master.py export --checkpoint stage2_perceptual.pth --output_dir deployment/exported
```

## 6. Current System Status

| Component | File | Status |
|-----------|------|--------|
| **Complexity Mask** | `src/core/complexity_mask.py` | ✅ **Functional** (Fourier-Variance) |
| **Geometric Engine** | `src/engines/geometric/surface_fit.py` | ✅ **Functional** (Polynomial) |
| **Structural Engine** | `src/engines/structural/dictionary.py` | ✅ **Functional** (Codebook) |
| **Neural Engine** | `src/engines/neural/lightweight.py` | ✅ **Functional** (GhostNet) |
| **Hybrid Orchestrator** | `src/core/orchestrator.py` | ✅ **Functional** |
| **Entropy Engine** | `src/utils/entropy_coder.py` | ✅ **Functional** (Vectorized Arithmetic) |
| **Binary Packing** | `src/core/blueprint_format.py` | ✅ **Functional** (v2 .padox) |
| **Stage 1 Trainer** | `src/train/trainer_stage1.py` | ✅ **Functional** (Foundation) |
| **Stage 2 Trainer** | `src/train/trainer_stage2.py` | ✅ **Functional** (Perceptual) |
| **Stage 3 Trainer** | `src/train/trainer_stage3.py` | ✅ **Functional** (QAT) |
| **Mobile Export** | `deployment/mobile_export.py` | ✅ **Functional** (ONNX + INT8) |
| **Native Quantizer** | `deployment/int8_quantizer.py` | ✅ **Functional** (PyTorch PTQ) |
| **Mobile JIT** | `deployment/android_ios_jit.py` | ✅ **Functional** (TorchScript/.ptl) |
| **Master CLI** | `blueprint_master.py` | ✅ **Functional** |

---
**Tip:** If you are using Google Drive to store checkpoints, mount it first:
```python
from google.colab import drive
drive.mount('/content/drive')
```
And update the save path in the training scripts.
