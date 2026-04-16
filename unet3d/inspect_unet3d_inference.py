import os
import sys
import warnings
# Enable MPS fallback for Mac users automatically
if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
# Suppress the specific MPS fallback warning
warnings.filterwarnings("ignore", message="The operator 'aten::slow_conv3d_forward' is not currently supported on the MPS backend")
warnings.filterwarnings("ignore", message="The operator 'aten::poisson' is not currently supported on the MPS backend")
warnings.filterwarnings("ignore", message="The provided colors parameter contained illegal values")

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from hologram_dataset import HologramDataset
from unet3d_model import ConcatDeepResUNet3D
from losses import BalancedHybridLoss, particles_mse, compute_distinct_centers
from torchsummary import summary
from pathlib import Path
parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)
from utils.holo_utils import compute_3d_metrics
import napari

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")


MODEL_PATH = Path(__file__).parent / "unet3d_dih_256_1.pth" # unet3d_dih_256_0.pth	unet3d_dih_256_1.pth	unet3d_dih_64_0.pth	unet3d_dih_64_1.pth


PEAK_THRESHOLD_PROB = 0.7
CONF_THRESHOLD = 0.7
DIST_THRESHOLD = 3.0


def inspect_one_sample(model):

    MIN_PARTICLES = 1
    MAX_PARTICLES = 100

    # GET a sample with random number of particles
    #particles =  torch.randint(MIN_PARTICLES, MAX_PARTICLES, (1,)).item()

    particles = 60

    test_dataset = HologramDataset(num_samples=1, num_particles=particles, is_train=False)

    dirty_vol, target_vol, hologram_2d, gt_coords = test_dataset[0]
    # print('hologram_2d.shape: ', hologram_2d.shape)

   # PERFORM INFERENCE
    with torch.no_grad():
        # Add batch dimension
        input_tensor = dirty_vol.unsqueeze(0).to(DEVICE)
        logits = model(input_tensor)
        prediction = torch.sigmoid(logits).squeeze(0).squeeze(0)

        pred_points, confs = compute_distinct_centers(logits, num_points=MAX_PARTICLES, peak_threshold_prob=PEAK_THRESHOLD_PROB)
        pred_points = pred_points[0][confs[0] > CONF_THRESHOLD]

        print(f"Ground truth {particles} particles.")
        print(f"Found {len(pred_points)} particles.")
        print(f"MSE: {particles_mse(gt_coords.to(DEVICE), pred_points) }")


    # PREPARE DATA FOR NAPARI
    # Convert tensors to numpy arrays
    pred_points_np = pred_points.cpu().numpy()
    asm_volume = dirty_vol[0].cpu().numpy()
    gt_volume = target_vol[0].cpu().numpy()
    pred_volume = prediction.cpu().numpy()
    source_hologram = hologram_2d.cpu().numpy()
    gt_coords_np = gt_coords.cpu().numpy()

    # LAUNCH NAPARI VIEWER
    viewer = napari.Viewer(title="DIH Particle Tracking - Final Test")

    # Add 2D Source Hologram (The raw interference pattern)
    viewer.add_image(source_hologram, name='1. Source 2D Hologram', colormap='viridis', blending='additive')

    # Add ASM Volume (Input to U-Net)
    viewer.add_image(asm_volume, name='2. Dirty Input (ASM Volume)', colormap='gray', blending='additive')

    # Add Ground Truth Target (Gaussian Blobs)
    viewer.add_image(gt_volume, name='3. Ground Truth Mask', colormap='green', blending='additive', opacity=0.5)

    # Add the Raw GT Coordinates as points
    arm = 3  # Total length of each line in the cross
    gt_lines = []
    for p in gt_coords_np:
        z, y, x = p
        # Line along Z axis
        gt_lines.append([[z - arm, y, x], [z + arm, y, x]])
        # Line along Y axis
        gt_lines.append([[z, y - arm, x], [z, y + arm, x]])
        # Line along X axis
        gt_lines.append([[z, y, x - arm], [z, y, x + arm]])
    viewer.add_shapes(gt_lines, shape_type='line', edge_width=0.1, edge_color='green',name='4. GT Coordinates', blending='additive')


    # Add U-Net Prediction (Denoised Volume)
    viewer.add_image(pred_volume, name='5. U-Net Prediction', colormap='magma', blending='additive')

    # Add Predicted Centroids (Points extracted from prediction)
    if len(pred_points_np) > 0:
        viewer.add_points(pred_points_np, size=1.0, name='6. Predicted Centroids', face_color='blue', border_color='blue ')


    print("\nVisualizing in Napari...")
    print("Toggle 3D view (cube icon) to see the axial reconstruction.")
    napari.run()


if __name__ == "__main__":

    # LOAD THE TRAINED MODEL
    model = ConcatDeepResUNet3D().to(DEVICE)

    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        #summary(model, input_size=(1, 64, 256, 256))
        model.eval()
        #print(f"Successfully loaded model from {MODEL_PATH}")

        inspect_one_sample(model)

    except FileNotFoundError:
        print(f"Error: {MODEL_PATH} not found. Please train the model first.")
