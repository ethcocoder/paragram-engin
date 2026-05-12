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

## 2. Training Stage 1: The Foundation
Stage 1 trains the **Structural Dictionary** (Pattern Memory) and the **Neural Engine** (Mobile Specialist) while keeping the Math Wizard and Complexity Mask frozen.

### Prepare a Real-Life HD Dataset
To test with real-world detail (not random noise), run this script to download 10 HD images:
```python
!python dataset_downloader.py
```

### Run the Trainer
Execute the Stage 1 trainer:
```python
!python src/train/trainer_stage1.py dataset/flickr8k
```

## 3. Current System Status
We are building the system in phases. Here is what is currently functional:

| Component | File | Status |
|-----------|------|--------|
| **Complexity Mask** | `src/core/complexity_mask.py` | ✅ **Functional** (Fourier-Variance) |
| **Math Wizard** | `src/engines/geometric/surface_fit.py` | ✅ **Functional** (Polynomial) |
| **Pattern Memory** | `src/engines/structural/dictionary.py` | ✅ **Functional** (Codebook) |
| **Mobile Specialist** | `src/engines/neural/lightweight.py` | ✅ **Functional** (GhostNet) |
| **Hybrid Brain** | `src/core/orchestrator.py` | ✅ **Functional** (Routing) |
| **Training Pipeline** | `src/train/trainer_stage1.py` | ✅ **Functional** |
| **Mobile Bridge** | `deployment/` | 🚧 *Docstrings only* |
| **Binary Packing** | `src/core/blueprint_format.py` | 🚧 *Docstrings only* |

## 4. Verification
After the first epoch, check for `reconstruction_check.png` in the file explorer. This shows the orchestrator's ability to reassemble images from the three hybrid paths.

---
**Tip:** If you are using Google Drive to store checkpoints, mount it first:
```python
from google.colab import drive
drive.mount('/content/drive')
```
And update the save path in `trainer_stage1.py`.
