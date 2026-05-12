# 🚀 Aether-Blueprint v3.0: Google Colab Deployment Guide

This guide will help you run the **Aether-Blueprint v3.0** training and inference pipeline on Google Colab to leverage the **Tesla T4 GPU**.

## 1. Environment Setup
Open a new notebook in Google Colab and set the runtime to **GPU** (`Runtime` -> `Change runtime type` -> `T4 GPU`).

### Clone / Upload the Code
If you have the code in a ZIP or on GitHub, upload it to the Colab environment. 
Alternatively, you can run this in a cell to initialize the structure:

```python
# Create the directory structure
!mkdir -p aether-blueprint-v3/{assets,configs,deployment,src/{core,engines/{geometric,neural,structural},utils,train},tests,dataset/flickr8k}
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
# Download the helper script
!curl -O https://raw.githubusercontent.com/USER/REPO/main/aether-blueprint-v3/dataset_downloader.py # If hosted
# OR just run the local script if uploaded:
!python aether-blueprint-v3/dataset_downloader.py
```

### Run the Trainer
Execute the Stage 1 trainer:
```python
%env PYTHONPATH=aether-blueprint-v3
!python aether-blueprint-v3/src/train/trainer_stage1.py dataset/flickr8k
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
