import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.fft as fft
import numpy as np

# --- PHYSICAL CONSTANTS ---
## SMALL GRID
"""
GRID_SIZE = 64
Z_SIZE = 64
PIXEL_SIZE = 3.45e-6
WAVELENGTH = 532e-9
Z_TOTAL = 10*Z_SIZE* PIXEL_SIZE # depth
Z_START, Z_STEP = 0.01, Z_TOTAL/Z_SIZE # physical units m, Z_START:  the distance between the particle and the sensor is between 1mm and 50mm
MIN_PARTICLES = 1
MAX_PARTICLES = 20
"""


## BIGGER GRID
GRID_SIZE = 256
Z_SIZE = 64
PIXEL_SIZE = 3.45e-6
WAVELENGTH = 532e-9
Z_TOTAL = 10*Z_SIZE*PIXEL_SIZE # depth
Z_START, Z_STEP = 0.01, Z_TOTAL/Z_SIZE # physical units m, Z_START:  the distance between the particle and the sensor is between 1mm and 50mm
MIN_PARTICLES = 1
MAX_PARTICLES = 100

# Use GPU if possible
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")


def propagate(field_2d, z_dist):
    """ASM propagation."""
    ny, nx = field_2d.shape
    fx = fft.fftfreq(nx, d=PIXEL_SIZE).to(DEVICE)
    fy = fft.fftfreq(ny, d=PIXEL_SIZE).to(DEVICE)
    FX, FY = torch.meshgrid(fx, fy, indexing='ij')
    k = 2 * np.pi / WAVELENGTH
    term = k**2 - (2*np.pi*FX)**2 - (2*np.pi*FY)**2
    phase = torch.sqrt(term.to(torch.complex64))
    kernel = torch.exp(1j * phase * z_dist)
    return fft.ifft2(fft.fft2(field_2d) * kernel)

class HologramDatasetClassic(Dataset):
    def __init__(self, num_samples=500, num_particles=60, is_train=True):
        if num_particles == 0:
            num_particles = torch.randint(MIN_PARTICLES, MAX_PARTICLES, (1,)).item()


        self.num_samples = num_samples
        self.is_train = is_train
        self.num_particles = num_particles

        self.clean_coords = []
        self.clean_targets = []

        # Anisotropic Gaussian Kernel
        # sigma_z = 0.5 (sharp axially), sigma_xy = 1.0 (wider laterally)
        z, y, x = torch.meshgrid(torch.arange(3)-1, torch.arange(7)-3, torch.arange(7)-3, indexing='ij')
        self.gaussian_kernel = torch.exp(-(z**2/(2*0.5**2) + y**2/(2*1.0**2) + x**2/(2*1.0**2))).to(DEVICE)

        for _ in range(num_samples):
            # Generate random 3D points
            batch_coords = []
            target_vol = torch.zeros((1, Z_SIZE, GRID_SIZE, GRID_SIZE), device=DEVICE)

            for _ in range(num_particles):
                zi = np.random.randint(5, Z_SIZE-5)
                yi, xi = np.random.randint(5, GRID_SIZE-5, size=2)
                batch_coords.append([zi, yi, xi])

                # Place GT Blob
                target_vol[0, zi-1:zi+2, yi-3:yi+4, xi-3:xi+4] = torch.maximum(
                    target_vol[0, zi-1:zi+2, yi-3:yi+4, xi-3:xi+4], self.gaussian_kernel)

            self.clean_coords.append(torch.tensor(batch_coords).float())
            self.clean_targets.append(target_vol)

    def __len__(self):
        return self.num_samples

    def apply_noise(self, h_intensity):
        """Applies physical noise models to the 2D intensity."""
        # Beam Profile (Gaussian laser profile)
        Y, X = torch.meshgrid(torch.linspace(-1, 1, GRID_SIZE, device=DEVICE),
                              torch.linspace(-1, 1, GRID_SIZE, device=DEVICE), indexing='ij')
        h_intensity *= torch.exp(-(X**2 + Y**2) / 4.0)

        # Randomized Noise Levels (Dynamic!)
        speckle_lvl = np.random.uniform(0, 0.08) if self.is_train else 0.01
        shot_lvl = np.random.choice([1000, 5000, 10000]) if self.is_train else 10000
        read_lvl = np.random.uniform(0, 0.01) if self.is_train else 0.001

        # Multiplicative Speckle
        h_intensity *= (1.0 + speckle_lvl * torch.randn_like(h_intensity))

        # Poisson Shot Noise
        h_intensity = torch.poisson(torch.clamp(h_intensity * shot_lvl, min=0)) / shot_lvl

        # Additive Read Noise
        h_intensity += read_lvl * torch.randn_like(h_intensity)

        return h_intensity

    def __getitem__(self, idx):
        coords = self.clean_coords[idx]
        target_vol = self.clean_targets[idx]

        # Build the clean 2D Field from pre-saved coords on the fly
        h_field = torch.ones((GRID_SIZE, GRID_SIZE), device=DEVICE, dtype=torch.complex64)
        for p in coords:
            zi, yi, xi = p.long()
            p_field = torch.zeros((GRID_SIZE, GRID_SIZE), device=DEVICE, dtype=torch.complex64)
            p_field[yi, xi] = 100
            h_field += propagate(p_field, Z_START + (zi * Z_STEP))

        # Get 2D Intensity and apply noise
        h_intensity = torch.abs(h_field)**2
        h_noisy = self.apply_noise(h_intensity)

        # Back-propagate noisy 2D to create 3D dirty volume
        dirty_vol = torch.zeros((1, Z_SIZE, GRID_SIZE, GRID_SIZE), device=DEVICE)
        h_complex = torch.sqrt(torch.clamp(h_noisy, min=0)).to(torch.complex64)
        for zi in range(Z_SIZE):
            dirty_vol[0, zi] = torch.abs(propagate(h_complex, -(Z_START + (zi * Z_STEP))))

        # dtandardize volume
        dirty_vol = (dirty_vol - dirty_vol.mean()) / (dirty_vol.std() + 1e-8)

        return dirty_vol, target_vol, h_noisy, coords

