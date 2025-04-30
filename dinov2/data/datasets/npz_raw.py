import torch
from torch.utils.data import Dataset
import numpy as np
import glob
import os
import torch.nn.functional as F

def downsample_raw(raw):
    """
    Downsamples a 4-channel packed RAW image by a factor of 2.
    The input raw should be a [H/2, W/2, 4] tensor -- with respect to its mosaiced version [H, W]
    Output is a [H/4, W/4, 4] tensor, preserving the RGGB pattern.
    """

    raw_image_4channel = torch.tensor(raw).unsqueeze(0).permute(0, 3, 1, 2)
    downsampled_image = F.avg_pool2d(raw_image_4channel, kernel_size=2, stride=2, padding=0)
    downsampled_image = downsampled_image.squeeze(0).permute(1, 2, 0)
    
    return downsampled_image

class NPZDataset(Dataset):
    def __init__(self, root, transform=None, target_transform=None, downsample=False):
        """
        Args:
            directory (string): Path to the directory containing .npz files.
            transform (callable, optional): Optional transform to be applied on a sample.
            target_transform (callable, optional): Optional transform to be applied on the target.
            downsample (bool, optional): If True, applies downsampling to the raw images.
        """
        self.file_paths = sorted(glob.glob(os.path.join(root, '*.npz')))
        self.transform = transform
        self.target_transform = target_transform
        self.downsample = downsample

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        npz_path = self.file_paths[idx]
        with np.load(npz_path) as data:
            raw_img = data["raw"]
            raw_max = data["max_val"]
            raw_img = (raw_img / raw_max).astype(np.float32)

        if self.downsample:
            raw_img = downsample_raw(raw_img)  # Apply downsampling

        if self.transform:
            raw_img = self.transform(raw_img)

        target = None
        if self.target_transform:
            target = self.target_transform(target)

        return raw_img, target

    def get_targets(self):
        return [None] * len(self.file_paths)
