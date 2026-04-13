import torch
import torch.nn.functional as F
import numpy as np
import napari
import time

# Setup device
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
if torch.cuda.is_available(): device = torch.device("cuda")
print(f"Using device: {device}")

def get_asm_tf(shape, z, wavelength, pixel_size, device):
    ny, nx = shape
    fx = torch.fft.fftfreq(nx, d=pixel_size, device=device)
    fy = torch.fft.fftfreq(ny, d=pixel_size, device=device)
    FX, FY = torch.meshgrid(fx, fy, indexing='xy')
    k = 2 * np.pi / wavelength
    h_arg = 1 - (wavelength**2 * (FX**2 + FY**2))
    H = torch.exp(1j * k * z * torch.sqrt(torch.clamp(h_arg, min=0)))
    return H * (h_arg > 0)

# Simulation physical setup
N = 512 # grid pixel size
pixel_size = 3.45e-6 # m
wavelength = 532e-9 # m
z_total = 0.02 # m

# particle number in a hologram
num_points_gt = 100

# Ground truth centers
gt_z_frac = torch.rand(num_points_gt, device=device) * 0.8 + 0.1
gt_y = torch.rand(num_points_gt, device=device) * (N - 120) + 60
gt_x = torch.rand(num_points_gt, device=device) * (N - 120) + 60
particles_gt_raw = torch.stack([gt_z_frac, gt_y, gt_x], dim=1)

def simulate_hologram(size):
    """Simulates complex field and hologram."""
    u_obj_sensor = torch.zeros((N, N), dtype=torch.complex64, device=device)

    # Calculate the physical thickness (t) of the cube based on pixel size
    # If size=10 pixels, and pixel_size=3.45um, then t = 34.5um
    t = size * pixel_size

    half = size // 2

    # The phase shift for the specific thickness
    delta_n = 0.05  # Assuming a refractive index difference of 0.05
    phase_shift = (2 * np.pi / wavelength) * delta_n * t

    # Pre-calculate the complex scattered field value
    # Use -1.0 so 'temp' only represents the scattered light
    scattered_field_value = torch.exp(torch.tensor(1j * phase_shift, device=device)) - 1.0

    for i in range(num_points_gt):
        p = particles_gt_raw[i]
        temp = torch.zeros((N, N), dtype=torch.complex64, device=device)
        y_s, y_e = int(p[1]) - half, int(p[1]) - half + size
        x_s, x_e = int(p[2]) - half, int(p[2]) - half + size

        # Inject the scattered phase field into the square
        temp[y_s:y_e, x_s:x_e] = scattered_field_value

        dist = p[0] * z_total
        tf = get_asm_tf((N, N), dist, wavelength, pixel_size, device)
        u_obj_sensor += torch.fft.ifft2(torch.fft.fft2(temp) * tf)

    # Off-axis tilt, assuming fixed physical setup
    shift_y, shift_x = 90, 120
    u_obj_fft = torch.fft.fft2(u_obj_sensor)
    u_obj_tilted_fft = torch.roll(u_obj_fft, shifts=(shift_y, shift_x), dims=(0, 1))

    # The complex wave at the sensor (after tilt)
    u_complex = torch.fft.ifft2(u_obj_tilted_fft)

    # The recorded hologram (Intensity of Complex Wave + Reference Wave)
    # Here Reference Wave is 1.0
    holo = torch.abs(u_complex + 1.0)**2

    return holo, u_complex, (shift_y, shift_x)

def evaluate_params(hologram, tilt, n_slices, thresh_mult, p_size):
    sy, sx = tilt
    dz = z_total / n_slices

    H_fft = torch.fft.fft2(hologram)
    fy_idx = torch.fft.fftfreq(N, device=device) * N
    fx_idx = torch.fft.fftfreq(N, device=device) * N
    FYI, FXI = torch.meshgrid(fy_idx, fx_idx, indexing='ij')
    mask = (torch.sqrt((FYI - sy)**2 + (FXI - sx)**2) < 60).float()
    H_base = torch.roll(H_fft * mask, shifts=(-sy, -sx), dims=(0, 1))

    vol_3d = torch.zeros((n_slices, N, N), device=device)
    for s in range(n_slices):
        dist_back = - (s * dz)
        tf_back = get_asm_tf((N, N), dist_back, wavelength, pixel_size, device)
        vol_3d[s] = torch.abs(torch.fft.ifft2(H_base * tf_back))

    threshold = (vol_3d.max() - vol_3d.min()) * thresh_mult + vol_3d.min()
    vol_input = vol_3d.unsqueeze(0).unsqueeze(0)
    k_size = max(7, p_size + 2)
    if k_size % 2 == 0: k_size += 1

    max_pool = F.max_pool3d(vol_input, kernel_size=k_size, stride=1, padding=k_size//2).squeeze()
    peaks_mask = (vol_3d == max_pool) & (vol_3d > threshold)
    coords = torch.nonzero(peaks_mask).to(torch.float32)

    num_det = coords.shape[0]
    if num_det == 0: return 1e9, 0, 1e9, None, vol_3d

    # Sub-voxel precision refinement
    z_idx = coords[:, 0].long()

    ## Boundary checking to filter out the very first slice  and very last slice
    valid = (z_idx > 0) & (z_idx < n_slices - 1)
    zv, yv, xv = z_idx[valid], coords[valid, 1].long(), coords[valid, 2].long()

    ## Intensities sample at three different depths
    y0, y1, y2 = vol_3d[zv-1, yv, xv], vol_3d[zv, yv, xv], vol_3d[zv+1, yv, xv]

    ## 3-point parabolic interpolation
    coords[valid, 0] = zv.float() + (y0 - y2) / (2 * (y0 - 2*y1 + y2) + 1e-12)

    curr_gt = particles_gt_raw.clone()
    curr_gt[:, 0] *= (n_slices - 1)

    # Convert detected coordinates to physical space in micrometers (* 1e6)
    coords_physical = coords.clone()
    coords_physical[:, 0] = (coords[:, 0] / (n_slices - 1)) * z_total * 1e6  # Z in µm
    coords_physical[:, 1] = coords[:, 1] * pixel_size * 1e6                  # Y in µm
    coords_physical[:, 2] = coords[:, 2] * pixel_size * 1e6                  # X in µm

    # Convert Ground Truth to physical space in MICROMETERS (* 1e6)
    curr_gt = particles_gt_raw.clone()
    curr_gt[:, 0] = curr_gt[:, 0] * z_total * 1e6                            # Z in µm
    curr_gt[:, 1] = curr_gt[:, 1] * pixel_size * 1e6                         # Y in µm
    curr_gt[:, 2] = curr_gt[:, 2] * pixel_size * 1e6                         # X in µm

    # Calculate true physical distances (in µm)
    dists = torch.cdist(curr_gt, coords_physical, p=2)
    min_dist_per_gt, _ = torch.min(dists, dim=1)

    # Calculate Root Mean Square Error (in µm)
    rmse_um = torch.sqrt(torch.mean(min_dist_per_gt**2)).item()

    # Penalize missed/extra detections
    # Using RMSE makes tuning this penalty weight (e.g., 5.0) much more intuitive
    score = rmse_um + (5.0 * abs(num_det - num_points_gt))

    return score, num_det, rmse_um, coords, vol_3d, dz*1e6

# Optimization parameter ranges-
size_options = np.arange(1, 10, 2)
slice_options = np.arange(100, 1600, 100)
threshold_options = np.arange(0.5, 0.9, 0.05)


best_global_score = float('inf')
best_global_results = {}

print(f"{'Size':<5} | {'Slices':<6} | {'Thresh':<6} | {'Detect':<6} | {'RMSE':<10} | {'Score'}")
print("-" * 60)

for ps in size_options:
    holo, u_complex, tilt = simulate_hologram(ps)

    for ns in slice_options:
        for tm in threshold_options:
            score, n_det, rmse_um, det_coords, vol, dz = evaluate_params(holo, tilt, ns, tm, ps)
            print(f"{ps:<5} | {ns:<6} | {tm:<6.2f} | {n_det:<6} | {rmse_um:<10.4f} | {score:.2f}")

            if score < best_global_score:
                best_global_score = score
                best_global_results = {
                    'ps': ps,
                    'ns': ns,
                    'tm': tm,
                    'n_det': n_det,
                    'rmse_um': rmse_um,
                    'dz': dz,
                    'vol': vol.cpu().numpy(), 'det': det_coords.cpu().numpy(),
                    'gt': particles_gt_raw.clone(),
                    'holo': holo.cpu().numpy(),
                    'mag': torch.abs(u_complex).cpu().numpy(),
                    'phase': torch.angle(u_complex).cpu().numpy()
                }
                best_global_results['gt'][:, 0] *= (ns - 1)

print("\n" + "="*40)
print("FINAL OPTIMAL SETTINGS")
print(f"Particle Size:  {best_global_results['ps']}x{best_global_results['ps']}x{best_global_results['ps']}")
print(f"Slice Count:    {best_global_results['ns']}")
print(f"Threshold Mult: {best_global_results['tm']}")
print(f"Detected:       {best_global_results['n_det']} / {num_points_gt}")
print(f"Final RMSE:      {best_global_results['rmse_um']:.6f}")
print(f"Final dz:      {best_global_results['dz']:.6f}")
print("="*40)

# Visualization
viewer = napari.Viewer()

# 2D Sensor Images
viewer.add_image(best_global_results['holo'], name="Hologram (Intensity)", colormap="gray", blending='additive')
viewer.add_image(best_global_results['mag'], name="Sensor Magnitude", colormap="magma", blending='additive', visible=False)
viewer.add_image(best_global_results['phase'], name="Sensor Phase", colormap="hsv", blending='additive', visible=False)

# 3D Reconstructed Volume
viewer.add_image(best_global_results['vol'], name="Best Recon Volume", colormap="viridis", rendering='mip')

# Points
viewer.add_points(best_global_results['gt'].cpu().numpy(), name="GT Particles",
                  face_color='transparent', border_color='magenta', symbol='square', size=5)
viewer.add_points(best_global_results['det'], name="Detected Peaks",
                  face_color='transparent', border_color='lime', symbol='+', size=5)

viewer.dims.ndisplay = 3
napari.run()
