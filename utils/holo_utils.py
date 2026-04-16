import torch
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment


def extract_2d_peaks(heatmap, threshold=0.10):
    # Use a very tight kernel (3x3) to allow close particles
    local_max = F.max_pool2d(heatmap, kernel_size=3, stride=1, padding=1)
    peak_mask = (heatmap == local_max) & (heatmap > threshold)

    # Add a local gradient check
    # Only keep the peak if it's a true "spike" and not just a flat plateau
    avg_pool = F.avg_pool2d(heatmap, kernel_size=3, stride=1, padding=1)
    spike_mask = heatmap > (avg_pool * 1.1) # Must be 10% brighter than its neighborhood

    final_mask = peak_mask & spike_mask
    return torch.nonzero(final_mask.squeeze()).float()

def extract_peaks(heatmap, threshold=0.1, kernel_size=3):
    """
    Performs 2D Non-Maximum Suppression to find local peaks in the heatmap.
    """
    # Max pool to find local maxima
    pad = kernel_size // 2
    local_max = F.max_pool2d(heatmap, kernel_size=kernel_size, stride=1, padding=pad)

    # Peaks are pixels that are equal to the local max and above threshold
    peak_mask = (heatmap == local_max) & (heatmap > threshold)

    # Extract indices
    coords = torch.nonzero(peak_mask.squeeze()) # [N, 2] -> (y, x)
    confs = heatmap[peak_mask]

    return coords, confs

def compute_metrics(pred_coords, gt_coords, dist_threshold=3.0):
    """
    Calculates TP, FP, FN and localization error using Hungarian Matching.
    """
    if len(pred_coords) == 0:
        return 0, 0, len(gt_coords), 0.0

    if len(gt_coords) == 0:
        return 0, len(pred_coords), 0, 0.0

    # Calculate distance matrix [N_pred, N_gt]
    # pred: (y, x), gt: (z, y, x) -> we take [:, 1:] for gt
    gt_coords = gt_coords[:, 1:]

    dists = torch.cdist(pred_coords.float(), gt_coords.float(), p=2).cpu().numpy()

    # (Hungarian Algorithm)
    row_ind, col_ind = linear_sum_assignment(dists)

    tp = 0
    errors = []
    matched_gt = set()

    for r, c in zip(row_ind, col_ind):
        if dists[r, c] < dist_threshold:
            tp += 1
            errors.append(dists[r, c])
            matched_gt.add(c)

    fp = len(pred_coords) - tp
    fn = len(gt_coords) - tp
    avg_error = np.mean(errors) if errors else 0.0

    return tp, fp, fn, avg_error

def compute_3d_metrics(pred_coords, gt_coords, dist_threshold=3.0):
    total_tp, total_fp, total_fn = 0, 0, 0

    dist_mat = torch.cdist(pred_coords, gt_coords)
    row_ind, col_ind = linear_sum_assignment(dist_mat.cpu().numpy())

    matched_pred = set()
    matched_gt = set()

    errors = []
    for r, c in zip(row_ind, col_ind):
        d_3d = dist_mat[r, c].item()

        # Only count as True Positive if it's within threshold
        if d_3d < dist_threshold:
            total_tp += 1
            errors.append(d_3d)
            matched_pred.add(r)
            matched_gt.add(c)

    total_fp += (len(pred_coords) - len(matched_pred))
    total_fn += (len(gt_coords) - len(matched_gt))
    avg_error = np.mean(errors) if errors else 0.0
    return total_tp, total_fp, total_fn, avg_error
