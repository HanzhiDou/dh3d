import os
import sys
import warnings
# Enable MPS fallback for Mac users automatically
if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
# Suppress the specific MPS fallback warning
warnings.filterwarnings("ignore", message="The operator 'aten::slow_conv3d_forward' is not currently supported on the MPS backend")

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

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")


MODEL_PATH = Path(__file__).parent / "unet3d_dih_256_1.pth" # unet3d_dih_256_0.pth	unet3d_dih_256_1.pth	unet3d_dih_64_0.pth	unet3d_dih_64_1.pth


PEAK_THRESHOLD_PROB = 0.7
CONF_THRESHOLD = 0.7
DIST_THRESHOLD = 3.0


def eval(model, num_particles):

    MIN_PARTICLES, MAX_PARTICLES = num_particles

    GRID_SIZE = 256 # 64, 256, 512, 768
    NOISE_LEVEL = 1 # 0, 1, 2, 3, 4
    num_samples = 10 # 300 for performance benchmarking, 10 for a quick test
    test_dataset = HologramDataset(num_samples=num_samples, num_particles=0, grid_size=GRID_SIZE, min_particles=MIN_PARTICLES, max_particles=MAX_PARTICLES, is_train=False, noise_level=NOISE_LEVEL)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # PERFORM INFERENCE
    all_tp, all_fp, all_fn = 0, 0, 0
    all_errors = []
    all_mse = []
    with torch.no_grad():
        for i, (dirty_vol, target_vol, hologram_2d, gt_coords) in enumerate(test_loader):
            dirty_vol = dirty_vol.to(DEVICE)
            gt_coords = gt_coords.to(DEVICE)

            logits = model(dirty_vol)
            pred_probs = torch.sigmoid(logits).squeeze(0).squeeze(0)

            pred_points, confs = compute_distinct_centers(logits, num_points=MAX_PARTICLES, peak_threshold_prob=PEAK_THRESHOLD_PROB)
            pred_points = pred_points[0][confs[0] > CONF_THRESHOLD]

            # Metrics
            gt_pts = gt_coords[0]  # [N, 3] (z, y, x)
            # print(f"Ground truth {len(gt_pts)} particles.")
            # print(f"Found {len(pred_points)} particles.")
            # print(f"MSE: {particles_mse(gt_pts.to(DEVICE), pred_points) }")
            # print('pred_points: ', pred_points.shape)
            # print('gt_pts: ', gt_pts.shape)
            tp, fp, fn, err = compute_3d_metrics(pred_points, gt_pts, DIST_THRESHOLD)

            all_tp += tp
            all_fp += fp
            all_fn += fn
            mse = particles_mse(gt_pts.to(DEVICE), pred_points)
            all_mse.append(mse)
            if tp > 0: all_errors.append(err)

            # print(f"Detected: {len(pred_points)} | GT: {len(gt_pts)} | Error: {err:.2f} px | MSE: {mse:.4f}")

    # Final Report
    precision = all_tp / (all_tp + all_fp + 1e-8)
    recall = all_tp / (all_tp + all_fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    rmse = np.mean(all_errors) if all_errors else 0.0

    print("\n" + "="*30)
    print(f"UNET3D EVALUATION REPORT(Particles: {MIN_PARTICLES}-{MAX_PARTICLES})")
    print("-" * 30)
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"Avg Error: {rmse:.4f} pixels")
    print(f"MSE: {np.mean(all_mse):.4f}")
    # print("="*30)


if __name__ == "__main__":

    # LOAD THE TRAINED MODEL
    model = ConcatDeepResUNet3D().to(DEVICE)

    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        summary(model, input_size=(1, 64, 256, 256))
        model.eval()
        #print(f"Successfully loaded model from {MODEL_PATH}")

        particles_ranges = [[1, 20],
                    [21, 40],
                    [41, 60],
                    [61, 80],
                    [81, 100],
                    [101, 120],
                    [121, 160],
                    [161, 200],
                    [201, 300]]


        # particles_ranges = [[1, 5],
        #                     [6, 10],
        #                     [11, 15],
        #                     [16, 20]]


        # particles_ranges = [
        #                     [40, 120]]


        for num_particles in particles_ranges:
            eval(model, num_particles)

    except FileNotFoundError:
        print(f"Error: {MODEL_PATH} not found. Please train the model first.")
