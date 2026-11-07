import numpy as np
import torch
from util.helper import cartesian_mask

class BrainAmbientFlowMatchingDataset(torch.utils.data.Dataset):
    def __init__(self, index, accel, cf):
        self.index = index
        self.accel = accel
        self.cf = cf

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        z = np.load(self.index[i])
        k = torch.from_numpy(z["kspace"])                              # stored k-space (coils, H, W)

        # Step 1: acceleration, i.e. the first downsampling mask A (fixed per slice)
        A = cartesian_mask(k.shape[-1], self.accel, self.cf, seed=i)   # (1, 1, W)

        # Step 2: apply it once, giving the accelerated k-space y = A x1
        y = A[0] * k

        return y, A, i    