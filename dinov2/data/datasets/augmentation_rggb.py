import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms.functional as TF
from torchvision import transforms
import random

class AdaptiveRawNormalize(nn.Module):
    """
    Normalizes raw data adaptively by estimating black and white levels
    """
    def __init__(self, percentile_black=0.1, percentile_white=99.9, clip=True):
        super().__init__()
        self.percentile_black = percentile_black
        self.percentile_white = percentile_white
        self.clip = clip
        
    def forward(self, img):
        # Convert to numpy for percentile calculation if needed
        if isinstance(img, torch.Tensor):
            is_tensor = True
            device = img.device
            img_np = img.cpu().numpy()
        else:
            is_tensor = False
            img_np = img
            
        # Estimate black and white levels
        black_level = np.percentile(img_np, self.percentile_black)
        white_level = np.percentile(img_np, self.percentile_white)
        
        # Safety check to avoid division by zero
        if black_level == white_level:
            white_level = black_level + 1.0
            
        # Convert back to tensor if needed
        if is_tensor:
            img = img.to(device)
            black_level = torch.tensor(black_level, device=device)
            white_level = torch.tensor(white_level, device=device)
            
        # Normalize to [0,1]
        img = (img - black_level) / (white_level - black_level)
        
        # Clip to [0,1] if requested
        if self.clip:
            if is_tensor:
                img = torch.clamp(img, 0, 1)
            else:
                img = np.clip(img, 0, 1)
                
        return img

# Other classes remain the same as previous example
class RawRandomResizedCrop(nn.Module):
    """
    Random resized crop that preserves Bayer pattern by ensuring even-valued crops
    """
    def __init__(self, size, scale=(0.08, 1.0)):
        super().__init__()
        if isinstance(size, (tuple, list)):
            self.size = size
        else:
            self.size = (size, size)
        self.scale = scale
            
    def forward(self, img):
        # Ensure dimensions are even to preserve the Bayer pattern
        height, width = img.shape[-2], img.shape[-1]
        
        # Calculate target area
        area = height * width
        target_area = random.uniform(self.scale[0], self.scale[1]) * area
        
        # Calculate aspect ratio
        aspect_ratio = random.uniform(3/4, 4/3)
        w = int(round(np.sqrt(target_area * aspect_ratio)))
        h = int(round(np.sqrt(target_area / aspect_ratio)))
        
        # Ensure width and height are even to preserve Bayer pattern
        w = w - (w % 2)
        h = h - (h % 2)
        
        # Make sure dimensions are valid
        if w <= width and h <= height:
            # Calculate starting point (must be even to preserve pattern)
            i = random.randint(0, height - h)
            j = random.randint(0, width - w)
            i = i - (i % 2)
            j = j - (j % 2)
        else:
            # Fallback to central crop
            i, j, h, w = 0, 0, height, width
            
        # Crop
        img = img[..., i:i+h, j:j+w]
        
        # Resize to target size (ensuring even sizes)
        target_h, target_w = self.size
        target_h = target_h - (target_h % 2)
        target_w = target_w - (target_w % 2)
        
        # Use bicubic interpolation
        img = TF.resize(img, [target_h, target_w], interpolation=TF.InterpolationMode.BICUBIC)
        
        return img

class RawHorizontalFlip(nn.Module):
    """
    Horizontal flip that preserves Bayer pattern by swapping R and B
    """
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        
    def forward(self, img):
        if random.random() < self.p:
            # For RGGB pattern:
            # First flip horizontally
            img = torch.flip(img, [-1])
            
            # Then swap R and B channels to maintain pattern
            # This is specific to RGGB pattern - adjust as needed for other patterns
            if len(img.shape) == 2:  # Raw Bayer pattern as 2D array
                # Create a copy
                flipped = img.clone()
                
                # Swap R and B pixels
                # For RGGB, R is at even rows/even cols, B is at odd rows/odd cols
                # After horizontal flip, we need to correct pattern
                flipped[0::2, 0::2] = img[0::2, 1::2]  # R becomes G
                flipped[0::2, 1::2] = img[0::2, 0::2]  # G becomes R
                flipped[1::2, 0::2] = img[1::2, 1::2]  # G becomes B
                flipped[1::2, 1::2] = img[1::2, 0::2]  # B becomes G
                
                return flipped
            elif len(img.shape) == 3 and img.shape[0] == 4:  # Raw as 4 channels [R,G,G,B]
                # For stacked RGGB, just swap R and G, and G and B
                r, g1, g2, b = img[0], img[1], img[2], img[3]
                return torch.stack([b, g2, g1, r])
                
        return img

class RawGaussianBlur(nn.Module):
    """
    Gaussian blur that respects raw data structure
    """
    def __init__(self, kernel_size=3, sigma=(0.1, 2.0), p=0.5):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.p = p
        
    def forward(self, img):
        if random.random() < self.p:
            sigma = random.uniform(self.sigma[0], self.sigma[1])
            
            # If raw is 2D array, unpack to channels first
            if len(img.shape) == 2:
                h, w = img.shape
                # Extract RGGB channels
                r = img[0::2, 0::2]
                g1 = img[0::2, 1::2]
                g2 = img[1::2, 0::2]
                b = img[1::2, 1::2]
                
                # Blur each channel separately
                r_blur = TF.gaussian_blur(r.unsqueeze(0).unsqueeze(0), self.kernel_size, sigma).squeeze()
                g1_blur = TF.gaussian_blur(g1.unsqueeze(0).unsqueeze(0), self.kernel_size, sigma).squeeze()
                g2_blur = TF.gaussian_blur(g2.unsqueeze(0).unsqueeze(0), self.kernel_size, sigma).squeeze()
                b_blur = TF.gaussian_blur(b.unsqueeze(0).unsqueeze(0), self.kernel_size, sigma).squeeze()
                
                # Reconstruct the Bayer pattern
                result = img.clone()
                result[0::2, 0::2] = r_blur
                result[0::2, 1::2] = g1_blur
                result[1::2, 0::2] = g2_blur
                result[1::2, 1::2] = b_blur
                
                return result
            elif len(img.shape) == 3 and img.shape[0] == 4:
                # For 4-channel representation, blur each separately
                return torch.stack([
                    TF.gaussian_blur(img[0:1], self.kernel_size, sigma),
                    TF.gaussian_blur(img[1:2], self.kernel_size, sigma),
                    TF.gaussian_blur(img[2:3], self.kernel_size, sigma),
                    TF.gaussian_blur(img[3:4], self.kernel_size, sigma)
                ]).squeeze(1)
                
        return img

class LambdaModule(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd
        
    def forward(self, x):
        return self.lambd(x)

class RawRandomBrightness(nn.Module):
    """
    Adjust brightness while maintaining pattern
    """
    def __init__(self, brightness=0.2):
        super().__init__()
        self.brightness = brightness
        
    def forward(self, img):
        brightness_factor = random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
        return img * brightness_factor

def create_adaptive_raw_pipeline(global_crops_size, local_crops_size, 
                              global_crops_scale=(0.4, 1.0), 
                              local_crops_scale=(0.05, 0.4),
                              local_crops_number=8):
    """
    Create an augmentation pipeline that adapts to each raw image's characteristics
    """
    # Convert tensor if input is numpy array
    # to_tensor = transforms.Lambda(lambda x: torch.from_numpy(x).float() if isinstance(x, np.ndarray) else x)
    to_tensor = LambdaModule(lambda x: torch.from_numpy(x).float() if isinstance(x, np.ndarray) else x)
    # Geometric transformations for global crops
    geometric_augmentation_global = nn.Sequential(
        to_tensor,
        RawRandomResizedCrop(global_crops_size, scale=global_crops_scale),
        RawHorizontalFlip(p=0.5),
    )
    
    # Geometric transformations for local crops
    geometric_augmentation_local = nn.Sequential(
        to_tensor,
        RawRandomResizedCrop(local_crops_size, scale=local_crops_scale),
        RawHorizontalFlip(p=0.5),
    )
    
    # Use adaptive normalization
    normalize = AdaptiveRawNormalize()
    
    # Photometric augmentations
    global_transfo1_extra = RawGaussianBlur(p=1.0)
    
    global_transfo2_extra = nn.Sequential(
        RawGaussianBlur(p=0.1),
        RawRandomBrightness(),
    )
    
    local_transfo_extra = RawGaussianBlur(p=0.5)
    
    # Full transformations
    global_transfo1 = nn.Sequential(global_transfo1_extra, normalize)
    global_transfo2 = nn.Sequential(global_transfo2_extra, normalize)
    local_transfo = nn.Sequential(local_transfo_extra, normalize)
    
    # Final augmentation class
    class AdaptiveRawAugmentation:
        def __init__(self):
            self.geometric_augmentation_global = geometric_augmentation_global
            self.geometric_augmentation_local = geometric_augmentation_local
            self.global_transfo1 = global_transfo1
            self.global_transfo2 = global_transfo2
            self.local_transfo = local_transfo
            self.local_crops_number = local_crops_number
            
        def __call__(self, image):
            output = {}
            
            # Global crops
            im1_base = self.geometric_augmentation_global(image)
            global_crop_1 = self.global_transfo1(im1_base)
            
            im2_base = self.geometric_augmentation_global(image)
            global_crop_2 = self.global_transfo2(im2_base)
            
            output["global_crops"] = [global_crop_1, global_crop_2]
            output["global_crops_teacher"] = [global_crop_1, global_crop_2]
            
            # Local crops
            local_crops = [
                self.local_transfo(self.geometric_augmentation_local(image)) 
                for _ in range(self.local_crops_number)
            ]
            output["local_crops"] = local_crops
            output["offsets"] = ()
            
            return output
            
    return AdaptiveRawAugmentation()

# Dataset class with modifications for raw RGGB data
class RawUniformDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        """
        Dataset for raw RGGB image data with adaptive handling
        """
        super().__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        
        self.data = []
        self.label_map = {}
        self._load_dataset()
        
    def _load_dataset(self):
        """Loads image paths and labels, mapping text labels to numbers."""
        label_idx = 0

        for subfolder in sorted(os.listdir(self.root)):  
            subfolder_path = os.path.join(self.root, subfolder)
            images_folder = os.path.join(subfolder_path, "images")
            labels_file = os.path.join(subfolder_path, "labels.txt")

            if not os.path.isdir(images_folder):
                continue

            label_dict = {}

            if os.path.exists(labels_file):
                with open(labels_file, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split(" ")
                        if len(parts) < 2:
                            continue  # Skip malformed lines

                        filename, *label_text = parts
                        label_text = " ".join(label_text)

                        # Assign numeric label
                        if label_text not in self.label_map:
                            self.label_map[label_text] = label_idx
                            label_idx += 1

                        label_dict[filename] = self.label_map[label_text]

            # Collect image paths and labels
            for img_name in sorted(os.listdir(images_folder)):
                img_path = os.path.join(images_folder, img_name)
                label = label_dict.get(img_name, -1)  # -1 if no label

                self.data.append((img_path, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]

        # Load raw RGGB data
        image = np.load(img_path)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image)
            
            # For compatibility with existing code expecting a single image
            # You can modify this based on how you want to use the crops
            image = transformed["global_crops"][0]  # Use first global crop
            
            # Or return all crops if your model expects them
            # return transformed, label
        
        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def get_targets(self):
        """Returns a list of all labels in the dataset."""
        return [label for _, label in self.data]