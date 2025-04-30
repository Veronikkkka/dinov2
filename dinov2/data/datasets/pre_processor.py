import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF


class RAWDataPreProcessor:
    """Data pre-processor specifically for RGGB RAW images."""
    
    def __init__(self, 
                 mean=None, 
                 std=None,
                 black_level=None,
                 white_level=None):
        self.mean = mean
        self.std = std
        self.black_level = black_level
        self.white_level = white_level

    def process_tensor(self, tensor):
        """Process a single tensor with normalization."""

        max_value = 16383 if tensor.max() > 1 else 1    # max for Nikon D700
        white_level = max(self.white_level, max_value)
        if self.black_level is not None and self.white_level is not None:
            tensor = (tensor - self.black_level) / (white_level - self.black_level)

            tensor = torch.clamp(tensor, 0, 1)

        if self.mean is not None and self.std is not None:
            tensor = (tensor - self.mean) / self.std
            
        return tensor

    def __call__(self, data):
        """Process either a tensor or a dict containing tensors."""
        if isinstance(data, dict):
            if 'image' in data:
                data['image'] = self.process_tensor(data['image'])
            elif 'inputs' in data:
                data['inputs'] = self.process_tensor(data['inputs'])
            # Process other potential image keys
            for key in ['raw', 'tensor', 'data']:
                if key in data and torch.is_tensor(data[key]):
                    data[key] = self.process_tensor(data[key])
            return data
        elif torch.is_tensor(data):
            return self.process_tensor(data)
        else:
            return data

import ast

class UniformDataset(Dataset):
    def __init__(self, root, transform=None, target_transform=None, split="train",
                 mean=None, std=None, black_level=0, white_level=1.0):
        """
        Args:
            root (str): Path to the main dataset directory.
            transform (callable, optional): Transformations to apply to the images.
            target_transform (callable, optional): Transformations to apply to labels.
            mean (list, optional): Mean values for normalization.
            std (list, optional): Standard deviation values for normalization.
            black_level (int/float): Black level for RAW image normalization.
            white_level (int/float): White level for RAW image normalization.
        """
        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        
        self.preprocessor = RAWDataPreProcessor(
            mean=mean,
            std=std,
            black_level=black_level,
            white_level=white_level
        )

        self.data = []
        self.label_map = {}
        self._load_dataset()

    
    def _load_dataset(self):
        """Loads image paths and labels, mapping text labels to numbers."""
        label_idx = 0
        split_root = os.path.join(self.root, self.split)

        label_dict = {}
        labels_file = os.path.join(self.root, "generalized_labels.txt")
        if os.path.exists(labels_file):
            with open(labels_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split(" ")
                    if len(parts) < 2:
                        continue
                    filename, *label_text = parts

                    label_text = label_text[0].split(";")

                    if len(label_text) > 1:
                        if "indoor" in label_text:
                            label_text = "indoor"
                        elif "outdoor" in label_text:
                            label_text = "outdoor"
                        else:
                            continue
                    else:
                        label_text = label_text[0]
                    if label_text not in self.label_map:
                        self.label_map[label_text] = label_idx
                        label_idx += 1
                    label_dict[filename] = self.label_map[label_text]


        for subfolder in sorted(os.listdir(split_root)):  

            subfolder_path = os.path.join(split_root, subfolder)
            images_folder = os.path.join(subfolder_path, "images")
            

            if not os.path.isdir(images_folder):
                continue


            max = 0
            p = None
            i = 0
            for img_name in sorted(os.listdir(images_folder)):
                img_path = os.path.join(images_folder, img_name)
                base_name = img_name.split(".npy")[0]
                label = label_dict.get(base_name, None)
                if label == None:
                    continue
                if label > max:
                    max = label
                    p = img_path
                self.data.append((img_path, label))
                i += 1
        print("Max", p, max)
        self.check_after_loading()
        print(self.label_map)

    def check_after_loading(self):
        import random, time
        random.seed(time.time()) 
        print("Time: ", time.time())
        save_dir = "check_before_training"
        os.makedirs(save_dir, exist_ok=True)

        indices = random.sample(range(len(self.data)), min(10, len(self.data)))

        for i, idx in enumerate(indices):
            image_path, label = self.data[idx]
            loaded_data = np.load(image_path, allow_pickle=True)
            raw_image = loaded_data
            raw_image = raw_image.astype(np.float32) 
            
            raw_tensor = torch.from_numpy(raw_image).float()
            processed_dict = self.preprocessor({"image": raw_tensor})
            visualization_image = processed_dict['image'][:3]
            visualization_image = visualization_image.cpu().detach().numpy()
            visualization_image = visualization_image.transpose(1, 2, 0)
            visualization_image = (visualization_image * 255).clip(0, 255).astype(np.uint8)
            img = Image.fromarray(visualization_image)
            img_save_path = os.path.join(save_dir, f"image_{i}.png")
            img.save(img_save_path)

        print(f"Saved {len(indices)} images to '{save_dir}'")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]

        try:

            loaded_data = np.load(img_path, allow_pickle=True)
            if isinstance(loaded_data, dict):

                data_dict = loaded_data
                if 'image' in data_dict:
                    raw_image = data_dict['image']
                elif 'raw' in data_dict:
                    raw_image = data_dict['raw']
                elif 'data' in data_dict:
                    raw_image = data_dict['data']
                else:

                    raw_image = next(iter(data_dict.values()))
            elif isinstance(loaded_data, np.ndarray):
                raw_image = loaded_data
                data_dict = {'image': raw_image}
            else:
                print(f"Unexpected data type: {type(loaded_data)} for {img_path}")
                raw_image = loaded_data
                data_dict = {'image': raw_image}


            if isinstance(raw_image, np.ndarray):
                if raw_image.dtype == np.uint16:
                    raw_image = raw_image.astype(np.float32) 

                if len(raw_image.shape) == 3 and raw_image.shape[0] == 4:
                    # (4, H, W)
                    raw_tensor = torch.from_numpy(raw_image).float()
                elif len(raw_image.shape) == 2:

                    raw_tensor = torch.from_numpy(raw_image).float().unsqueeze(0)
                elif len(raw_image.shape) == 3 and raw_image.shape[2] in [1, 3, 4]:
                    raw_tensor = torch.from_numpy(raw_image.transpose(2, 0, 1)).float()
                else:
                    raw_tensor = torch.from_numpy(raw_image).float()
                    if len(raw_tensor.shape) == 2:
                        raw_tensor = raw_tensor.unsqueeze(0)
                
                data_dict['image'] = raw_tensor
                

            elif torch.is_tensor(raw_image):
                raw_tensor = raw_image.float()
                data_dict['image'] = raw_tensor

            processed_dict = self.preprocessor(data_dict)
            
            
            processed_image = processed_dict['image']

            processed_image = processed_image.cpu().detach()

            if self.transform:
                image = self.transform(processed_image)

            if type(image) != torch.Tensor:
                self.visualize_rggb_image(image['global_crops'][0].permute(1, 2, 0), "global_crop1.png")
            else:

                self.visualize_rggb_image(image.permute(1, 2, 0), "global_crop1.png")

            
            if self.target_transform and label is not None:
                label = self.target_transform(label)
            # processed_image.save("output.png")  



            return image, label
            
        except (EOFError, ValueError, Exception) as e:
            print(f"Error loading file {img_path}: {str(e)}")
            placeholder = torch.zeros((4, 224, 224))
            return placeholder, label

    
    def get_targets(self):
        """Returns a list of all labels in the dataset."""
        return np.array([label for _, label in self.data], dtype=np.int64)
    
    def visualize_rggb_image(self, tensor, output_path="visualized_image.png"):
        """
        Properly visualize a raw image tensor with shape [H, W, 4] where the last dimension is RGGB
        
        Args:
            tensor: Raw tensor with shape [H, W, 4]
            output_path: Path to save the visualization
        """

        if isinstance(tensor, torch.Tensor):
            if tensor.device.type != 'cpu':
                tensor = tensor.cpu()
            tensor_np = tensor.numpy()
        else:
            tensor_np = tensor
        
        r = tensor_np[:, :, 0]
        g1 = tensor_np[:, :, 1]
        g2 = tensor_np[:, :, 2]
        b = tensor_np[:, :, 3]
        
        g = (g1 + g2) / 2
        
        rgb = np.stack([r, g, b], axis=2)
        

        def robust_normalize(arr):
            lower = np.percentile(arr, 2)
            upper = np.percentile(arr, 98)
            
            if upper == lower:
                return np.zeros_like(arr)
            
            arr_clipped = np.clip(arr, lower, upper)
            arr_norm = (arr_clipped - lower) / (upper - lower)
            return np.clip(arr_norm * 255, 0, 255).astype(np.uint8)
        
        rgb_normalized = robust_normalize(rgb)

        Image.fromarray(rgb_normalized).save(output_path)
        
        return rgb_normalized

    def load_and_trivial_rggb_to_rgb(self, npy_path):
        import cv2
        array = np.load(npy_path)  #(4, H, W)
        if array.dtype != np.uint8:
            if array.max() > 1:
                array = (array / array.max() * 255).astype(np.uint8)
            else:
                array = (array * 255).astype(np.uint8)

        # print("After conversion:", array.min(), array.max(), array.dtype)
        # print(array)
        bayer_mosaic = pack_rggb_planes_to_bayer(array)
        # Demosaic to RGB
        rgb = cv2.cvtColor(bayer_mosaic, cv2.COLOR_BAYER_RG2RGB)

        raw_image = Image.fromarray(rgb)
        return raw_image

def pack_rggb_planes_to_bayer(raw_rggb_planes):
    """
    Takes a (4, H, W) array with RGGB planes and returns a (2H, 2W) Bayer mosaic image.
    Assumes:
      raw[0] = R
      raw[1] = G1
      raw[2] = G2
      raw[3] = B
    """
    R, G1, G2, B = raw_rggb_planes
    H, W = R.shape

    bayer = np.zeros((H * 2, W * 2), dtype=R.dtype)

    bayer[0::2, 0::2] = R     # Top-left
    bayer[0::2, 1::2] = G1    # Top-right
    bayer[1::2, 0::2] = G2    # Bottom-left
    bayer[1::2, 1::2] = B     # Bottom-right

    return bayer