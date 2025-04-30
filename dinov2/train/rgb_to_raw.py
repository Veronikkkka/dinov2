import numpy as np
import cv2
import argparse
from pathlib import Path
import torch
from PIL import Image
import numpy as np


def rgb_to_raw(image_path="", img=None):
    if img is None:
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    else:
        img = np.array(img)
    
    if img is None:
        raise ValueError("Error loading image. Please check the path.")

    if len(img.shape) == 3:
        img_raw = img[:, :, 1] 
    else:
        img_raw = img
    

    if img_raw.dtype != np.uint16:
        img_raw = (img_raw.astype(np.float32) / 255.0 * 65535).astype(np.uint16)
    
    return img_raw




import numpy as np
import cv2

def raw_to_rgb(raw_array, pattern='RGGB', image_size=(256, 256), bits=16):
    """
    Convert RAW sensor data to RGB image with improved handling of various bit depths
    and white balance correction.
    
    Args:
        raw_array: NumPy array containing RAW sensor data
        pattern: Bayer pattern ('RGGB', 'BGGR', 'GRBG', 'GBRG')
        image_size: Tuple of (height, width)
        bits: Bit depth of the RAW data (typically 12, 14, or 16)
    
    Returns:
        RGB image as numpy array
    """
    if not isinstance(raw_array, np.ndarray):
        raise TypeError("Input must be a NumPy array.")
    
   
    max_value = 2**bits - 1
    raw_image = raw_array
    if raw_image.dtype != np.uint8:
        raw_image = (raw_image / max_value * 255).astype(np.uint8)
    
    bayer_patterns = {
        'RGGB': cv2.COLOR_BayerBG2BGR,
        'BGGR': cv2.COLOR_BayerGB2BGR,
        'GRBG': cv2.COLOR_BayerGR2BGR,
        'GBRG': cv2.COLOR_BayerRG2BGR
    }
   
    rgb_image = cv2.demosaicing(raw_image, bayer_patterns[pattern])
    
    rgb_image_float = rgb_image.astype(float)
    for i in range(3):
        channel = rgb_image_float[:,:,i]
        mean_val = np.mean(channel)
        if mean_val > 0:
            channel *= 128 / mean_val
            rgb_image_float[:,:,i] = channel
    
    rgb_image = np.clip(rgb_image_float, 0, 255)
    pil_image = Image.fromarray(rgb_image.astype(np.uint8), mode='RGB')

    
    return pil_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert RGB image to RAW format.")
    parser.add_argument("image_path", type=str, help="Path to the input image")

    args = parser.parse_args()
    
    raw = rgb_to_raw(args.image_path)
    print(raw, type(raw), raw.shape)
    rgb = raw_to_rgb(raw, image_size=(512, 683))
    
    print(rgb)
