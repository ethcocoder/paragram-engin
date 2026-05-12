# 🛰️ Aether-Blueprint v3.0: Project Architectural Audit

This document provides a comprehensive overview of the current state of **Aether-Blueprint v3.0: The Quantum-Structural Hybrid**.

## 1. Project Mission
To build a world-class, mobile-ready image synthesis engine that utilizes **INT8 Quantization-Aware Training (QAT)** and a **Hybrid Routing** system to achieve high-fidelity compression on smartphone NPUs.

---

## 2. Completed Core Components (The "Active" Engine)

### 🧠 Complexity Mask (`src/core/complexity_mask.py`)
- **Logic**: Vectorized Fourier-Variance path.
- **Function**: Categorizes image tiles into three states:
    - **State 0 (Vacuum)**: Low energy/entropy (Gradients).
    - **State 1 (Texture)**: High periodicity (Repeating patterns).
    - **State 2 (Detail)**: High chaotic energy (Unique features/Faces).
- **Status**: ✅ Fully Functional & Tested.

### 📐 Geometric Engine (`src/engines/geometric/surface_fit.py`)
- **Logic**: 3rd-degree Polynomial Surface Fitting.
- **Function**: Represents smooth surfaces using only 10 coefficients per channel.
- **Fidelity**: MSE `2.54e-14` (Mathematical precision).
- **Status**: ✅ Fully Functional & Tested.

### 📚 Structural Engine (`src/engines/structural/dictionary.py`)
- **Logic**: Pattern Memory (Texture Codebook).
- **Function**: Subdivides tiles into 32x32 patches and matches them against a 512-entry codebook. Supports 4-way rotation and Affine (Gain/Bias) correction.
- **Status**: ✅ Fully Functional & Tested.

### ⚡ Neural Engine (`src/engines/neural/lightweight.py`)
- **Logic**: GhostNet-based Mobile Specialist.
- **Function**: Uses GhostConvs, H-Swish, and Swin-Window Attention. Optimized for < 0.5M parameters to fit mobile VRAM.
- **Status**: ✅ Fully Functional & Tested.

### 🎛️ Hybrid Orchestrator (`src/core/orchestrator.py`)
- **Logic**: Central Routing Switchboard.
- **Function**: Coordinates the Complexity Mask to route tiles and reassembles them using the `reconstruct` method.
- **Status**: ✅ Fully Functional & Tested.

---

## 3. Training Milestone: Stage 1 Complete
- **Objective**: Establish the structural foundation and initialize the codebook.
- **Results**: Successfully trained for 20 epochs on Google Colab T4.
- **Stats**: 
    - **Math Usage**: ~15.6%
    - **Structural Usage**: ~80.6%
    - **Neural Usage**: ~3.8%
- **Checkpoint**: `stage1_foundation.pth` saved.

---

## 4. Current File Inventory (The "Skeleton")

The following files are currently initialized as **Stubs/Docstrings** and represent the remaining work to reach v3.0 production status:

### Core & Infrastructure
- `blueprint_format.py`: 🚧 **The Binary Vault**. Needs bit-packing logic for `.padox` files.
- `blueprint_master.py`: 🚧 **The Entry Point**. Needs CLI implementation.

### Engine Extensions
- `gradient_gen.py`: 🚧 **Parametric Gradients**. Support for complex transition zones.
- `swin_tiny.py`: 🚧 **Advanced Detail**. Dedicated Swin-Transformer logic (currently inline in `lightweight.py`).
- `fractal.py`: 🚧 **Recursive Patterns**. Self-similar rule discovery for the structural engine.

### Utilities
- `tiling_v3.py`: 🚧 **Smart Tiler**. Needs Variable-Density logic (currently using fixed 128x128).
- `perceptual.py`: 🚧 **Human Eye Tuning**. Needs LPIPS and MS-SSIM implementation.
- `entropy_coder.py`: 🚧 **Secondary Compression**. Range coding for latents and coefficients.

### Deployment & Mobile Bridge
- `mobile_export.py`: 🚧 **Export Logic**. ONNX / CoreML / TFLite conversion.
- `int8_quantizer.py`: 🚧 **PTQ**. Post-training quantization and calibration.
- `android_ios_jit.py`: 🚧 **JIT Optimization**. TorchScript tuning for mobile NPUs.

### Training Phases
- `trainer_stage2.py`: 🚧 **Perceptual Refinement**. Unfreezing attention for visual fidelity.
- `trainer_stage3.py`: 🚧 **QAT**. Training with 8-bit quantization noise.

---

## 5. Next Strategic Move
The "Brain" is functional, but the "Body" (Binary Packing) and "Polish" (Perceptual Loss) are next. 

**Recommendation**: Move to **`src/utils/perceptual.py`** to prepare for Stage 2 training, ensuring the model "sees" the same way a human does.
