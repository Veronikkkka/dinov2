import os
import numpy as np
import torch
from tqdm import tqdm
from dinov2.data.datasets.pre_processor import RAWDataPreProcessor  # Adjust import path

INPUT_DIR = "/home/paperspace/Documents/nika_space/main_dataset/raise/images"
OUTPUT_DIR = "/home/paperspace/Documents/nika_space/main_dataset/raise_preprocessed/images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

preprocessor = RAWDataPreProcessor(
    mean=None,
    std=None,
    black_level=0,
    white_level=16383
)

for root, dirs, files in os.walk(INPUT_DIR):
    for fname in tqdm(files):
        if not fname.endswith(".npy"):
            continue

        input_path = os.path.join(root, fname)
        output_path = os.path.join(OUTPUT_DIR, fname)

        if os.path.exists(output_path):
            continue

        raw_data = np.load(input_path, allow_pickle=True)

        if isinstance(raw_data, dict):
            if 'image' in raw_data:
                raw_tensor = torch.from_numpy(raw_data['image']).float()
            elif 'raw' in raw_data:
                raw_tensor = torch.from_numpy(raw_data['raw']).float()
            else:
                raw_tensor = torch.from_numpy(next(iter(raw_data.values()))).float()
        elif isinstance(raw_data, np.ndarray):
            raw_tensor = torch.from_numpy(raw_data.astype(np.float32)).float()
        else:
            continue

        processed_tensor = preprocessor.process_tensor(raw_tensor)

 
        processed_array = processed_tensor.numpy()
        np.save(output_path, processed_array)
