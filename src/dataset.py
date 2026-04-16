import torch
import zarr
import numpy as np
from torch.utils.data import Dataset

class MitoDataset(Dataset):
    """
    Handles OpenOrganelle Zarr data.
    Loads 'raw' (for details) and 'labels' (for shape).
    """
    def __init__(self, zarr_path, mode='train', crop_size=64, num_points=2048):
        self.crop_size = crop_size
        self.num_points = num_points
        self.mode = mode
        self.zarr_path = zarr_path
        
        # Lazy loading Zarr
        try:
            self.store = zarr.open(zarr_path, mode='r')
            # NOTE: Update these paths based on the specific dataset structure
            self.raw = self.store['volumes/raw/s0']
            self.labels = self.store['volumes/labels/mito/s0'] 
        except:
            print(f"[Warning] Could not load Zarr at {zarr_path}. Using Dummy Data.")
            self.raw = None
            self.labels = None

        # Pre-calculated centers (Placeholder logic)
        self.centers = self._get_centers()

    def _get_centers(self):
        # In a real thesis, scan the volume to find mitochondrial centroids.
        # Here we return dummy indices for compilation.
        return [[100, 100, 100]] * 10 

    def __len__(self):
        return len(self.centers)

    def __getitem__(self, idx):
        # Dummy generator if file not found (allows code to run for demo)
        if self.raw is None:
            return self._get_dummy_item()

        cz, cy, cx = self.centers[idx]
        r = self.crop_size // 2
        
        # 1. Extract Crops
        raw_crop = self.raw[cz-r:cz+r, cy-r:cy+r, cx-r:cx+r]
        mask_crop = self.labels[cz-r:cz+r, cy-r:cy+r, cx-r:cx+r]
        
        # Normalize
        raw_tensor = torch.from_numpy(raw_crop.astype(np.float32) / 255.0).unsqueeze(0)
        mask_binary = (mask_crop > 0).astype(np.float32)
        
        # 2. Sample Points for Implicit Training
        points, occ = self._sample_points(mask_binary)
        
        return {
            "inputs": raw_tensor,   # Condition (EM Image)
            "points": points,       # Query coords
            "occupancies": occ,     # GT labels
            "raw_numpy": raw_crop   # For post-processing
        }

    def _sample_points(self, mask):
        # Sample points in [-1, 1]
        D, H, W = mask.shape
        points = np.random.rand(self.num_points, 3).astype(np.float32)
        
        # Trilinear interpolation or nearest neighbor for occupancy GT
        # Simplified: Nearest neighbor
        vox_coords = (points * np.array([D-1, H-1, W-1])).astype(int)
        occ = mask[vox_coords[:,0], vox_coords[:,1], vox_coords[:,2]]
        
        points_norm = (points - 0.5) * 2.0 # [-1, 1]
        return torch.from_numpy(points_norm), torch.from_numpy(occ).float()

    def _get_dummy_item(self):
        return {
            "inputs": torch.randn(1, 64, 64, 64),
            "points": torch.randn(2048, 3),
            "occupancies": torch.randint(0, 2, (2048,)).float(),
            "raw_numpy": np.random.rand(64, 64, 64)
        }