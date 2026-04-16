## 🚀 Quick Start
This project uses [uv](https://github.com/astral-sh/uv) for dependency management. You do not need to install Python or manage environments manually.

1. Install uv:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. Run the ASM back-propagation 3D reconstruction samples:
   ```bash
   uv run oah_asm_reconst.py
   ```
3. Train the unet3d for Digital In-line Holography (DIH):
   ```bash
   uv run unet3d/train_unet3d.py
   ```
4. Evaluate the performance of a trained unet3d model for DIH:
   ```bash
   unet3d/eval_unet3d.py
   ```
5. Inspect one inference sample in Napari viewer:
   ```bash
   uv run unet3d/inspect_unet3d_inference.py
   ```
