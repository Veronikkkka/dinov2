import torch
from torch.utils.data import DataLoader
import numpy as np
from dinov2.data.datasets import UniformDataset
def compute_mean_std(dataset):
    """Computes mean and standard deviation for a 4-channel RGGB dataset."""
    loader = DataLoader(dataset, batch_size=32, num_workers=4, shuffle=False)

    mean = torch.zeros(4)
    std = torch.zeros(4)
    total_pixels = 0

    for images, _ in loader:
        if isinstance(images, dict):  
            images = images['image']  # If using a dict, extract the image tensor
        
        images = images.view(images.size(0), 4, -1)  # Flatten to (Batch, Channels, N)
        
        batch_mean = images.mean(dim=[0, 2])  # Mean over batch & pixels
        batch_std = images.std(dim=[0, 2])  # Std over batch & pixels
        
        num_pixels = images.shape[0] * images.shape[2]  # Total pixel count
        mean += batch_mean * num_pixels
        std += batch_std * num_pixels
        total_pixels += num_pixels

    mean /= total_pixels
    std /= total_pixels

    return mean.tolist(), std.tolist()

# Usage:
dataset = UniformDataset(root="/home/paperspace/Documents/nika_space/main_dataset/", transform=None)  
mean, std = compute_mean_std(dataset)
print("Dataset Mean:", mean)
print("Dataset Std:", std)
