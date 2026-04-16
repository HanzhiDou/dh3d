import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BalancedHybridLoss(nn.Module):
    def __init__(self, scale_vector, chamfer_weight=0.1, pos_weight=100.0):
        super().__init__()
        # BCE with high pos_weight is the discovery engine
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
        self.cham_w = chamfer_weight
        self.scale_vector = scale_vector


    def forward(self, logits, target_mask, gt_coords):
        # Discovery loss (scale: ~0.1 to 1.0)
        if target_mask is None:
            return 1e6
        loss_bce = self.bce(logits, target_mask)

        # Refinement loss (Normalized)
        # Get predicted centers in range [0, 1]
        pred_centers = compute_training_centers(logits, grid_split=4)
        if len(pred_centers) == 0:
            return 1e6

        # Normalize ground gruth to [0, 1]
        GRID_SIZE = logits.shape[-1]
        gt_norm = gt_coords / GRID_SIZE

        # print('pred_centers: ', pred_centers.shape)
        # print('gt_norm: ', gt_norm.shape)

        # Compute distance matrix (Symmetric)
        diff = pred_centers.unsqueeze(2) - gt_norm.unsqueeze(1)
        dist_mat = torch.sqrt(torch.sum(diff**2, dim=-1) + 1e-8)

        # Distance from GT to nearest Pred (ensures all particles are found)
        d_gt_to_p = torch.mean(torch.min(dist_mat, dim=1)[0])

        # Distance from Pred to nearest GT (ensures no ghost particles)
        d_p_to_g = torch.mean(torch.min(dist_mat, dim=2)[0])

        loss_chamfer = d_gt_to_p + d_p_to_g

        # Weighted Sum
        # By normalizing, loss_chamfer will be ~0.1-0.5.
        # Multiply by self.cham_w to make it a sub-task of the BCE.
        return loss_bce + (self.cham_w * loss_chamfer)



def particles_mse(gt_coords, pred_coords):
    if gt_coords.shape[0] == 0 and pred_coords.shape[0] != 0:
        return 1e3
    if gt_coords.shape[0] != 0 and pred_coords.shape[0] == 0:
        return 1e3

    dists = torch.cdist(gt_coords, pred_coords, p=2)
    min_dist_per_gt, _ = torch.min(dists, dim=1)
    mse = torch.mean(min_dist_per_gt**2).item()
    return mse


def compute_distinct_centers(logits, num_points=10, peak_threshold_prob=0.2, window_size=5):
    """
    Extracts distinct particle centers using 3D Non-Maximum Suppression (NMS).
    """
    B, C, Z, Y, X = logits.shape
    probs = torch.sigmoid(logits)
    device = logits.device

    # 3D Non-Maximum Suppression
    # Use a MaxPool3d to find the maximum value in every 5x5x5 neighborhood.
    # A pixel is a peak only if its value is equal to the max in its neighborhood.
    local_max = F.max_pool3d(probs, kernel_size=5, stride=1, padding=2)
    peak_mask = (probs == local_max) & (probs > peak_threshold_prob)

    # Extract and sort peaks
    final_centers = []
    final_confs = []

    r = window_size // 2

    for b in range(B):
        # Find coordinates of all peaks in this batch item
        # coords shape: [N_peaks, 3] -> (z, y, x)
        coords = torch.nonzero(peak_mask[b, 0])

        if coords.shape[0] == 0:
            # Fallback if no particles found
            final_centers.append(torch.zeros((num_points, 3), device=device))
            final_confs.append(torch.zeros(num_points, device=device))
            continue

        # Get the probability values at these peak coordinates to sort them
        peak_vals = probs[b, 0, coords[:, 0], coords[:, 1], coords[:, 2]]
        sorted_idx = torch.argsort(peak_vals, descending=True)
        coords = coords[sorted_idx]
        peak_vals = peak_vals[sorted_idx]

        # Limit to the requested number of points
        coords = coords[:num_points]
        peak_vals = peak_vals[:num_points]

        # Sub-pixel Refinement (Soft-Argmax)
        batch_refined = []
        for i in range(coords.shape[0]):
            zi, yi, xi = coords[i]

            # Define window boundaries
            z_s, z_e = max(0, zi-r), min(Z, zi+r+1)
            y_s, y_e = max(0, yi-r), min(Y, yi+r+1)
            x_s, x_e = max(0, xi-r), min(X, xi+r+1)

            win = probs[b, 0, z_s:z_e, y_s:y_e, x_s:x_e]

            # Weighted average for sub-pixel precision
            # Marginals (summing out other dims) are faster than meshgrids
            mass = win.sum() + 1e-8

            # Z-centroid
            wz = torch.arange(z_s, z_e, device=device).float()
            ref_z = (win.sum(dim=(1, 2)) * wz).sum() / mass

            # Y-centroid
            wy = torch.arange(y_s, y_e, device=device).float()
            ref_y = (win.sum(dim=(0, 2)) * wy).sum() / mass

            # X-centroid
            wx = torch.arange(x_s, x_e, device=device).float()
            ref_x = (win.sum(dim=(0, 1)) * wx).sum() / mass

            batch_refined.append(torch.stack([ref_z, ref_y, ref_x]))

        # Pad with zeros if we found fewer than num_points
        while len(batch_refined) < num_points:
            batch_refined.append(torch.zeros(3, device=device))
            peak_vals = torch.cat([peak_vals, torch.zeros(1, device=device)])

        final_centers.append(torch.stack(batch_refined))
        final_confs.append(peak_vals[:num_points])

    return torch.stack(final_centers), torch.stack(final_confs)



def compute_training_centers(logits, grid_split=4):
    """
    A differentiable, grid-based soft-argmax for TRAINING.
    Divides the volume into cells and finds the mass-center of each.
    """
    probs = torch.sigmoid(logits)
    B, _, Z, Y, X = probs.shape
    device = logits.device
    S = grid_split # e.g., 4x4x4 grid = 64 cells

    # Create normalized coordinate grid [0, 1]
    z_range = torch.linspace(0, 1, Z, device=device)
    y_range = torch.linspace(0, 1, Y, device=device)
    x_range = torch.linspace(0, 1, X, device=device)
    grid_z, grid_y, grid_x = torch.meshgrid(z_range, y_range, x_range, indexing='ij')
    grid = torch.stack([grid_z, grid_y, grid_x], dim=-1) # [Z, Y, X, 3]

    # Reshape into cells
    dz, dy, dx = Z//S, Y//S, X//S
    probs_cells = probs.view(B, S, dz, S, dy, S, dx).permute(0, 1, 3, 5, 2, 4, 6).reshape(B, S**3, dz, dy, dx, 1)
    grid_cells = grid.view(S, dz, S, dy, S, dx, 3).permute(0, 2, 4, 1, 3, 5, 6).reshape(1, S**3, dz, dy, dx, 3)

    # Soft-Argmax per cell
    mass = torch.sum(probs_cells, dim=(2, 3, 4)) + 1e-8
    centers = torch.sum(probs_cells * grid_cells, dim=(2, 3, 4)) / mass

    return centers # [B, S^3, 3] in range [0, 1]
