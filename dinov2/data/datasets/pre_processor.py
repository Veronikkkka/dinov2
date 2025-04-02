import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

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


class UniformDataset(Dataset):
    def __init__(self, root, transform=None, target_transform=None, 
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
                            continue

                        filename, *label_text = parts
                        label_text = " ".join(label_text)


                        if label_text not in self.label_map:
                            self.label_map[label_text] = label_idx
                            label_idx += 1

                        label_dict[filename] = self.label_map[label_text]


            for img_name in sorted(os.listdir(images_folder)):
                img_path = os.path.join(images_folder, img_name)
                base_name = img_name.split(".npy")[0]
                label = label_dict.get(base_name, -1)

                self.data.append((img_path, label))
            self.check_after_loading()

    def check_after_loading(self):
        import random
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
            img_save_path = os.path.join(save_dir, f"image_{i}.png")
            Image.fromarray(visualization_image).save(img_save_path)

        print(f"Saved {len(indices)} images to '{save_dir}'")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]

        try:

            loaded_data = np.load(img_path, allow_pickle=True)
            # print("Loaded ", loaded_data.shape, img_path)
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
                    # print(f"Converting dtype of {img_path} from uint16 to float32.")
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
                # print("Raw tensor", raw_tensor)
                data_dict['image'] = raw_tensor
                # print("Data dict", data_dict['image'])

            # print("Img path ", img_path, data_dict['image'])
            processed_dict = self.preprocessor(data_dict)
            
            
            processed_image = processed_dict['image']
            # print("Processed image: ", processed_image)
            processed_image = processed_image.cpu().detach()
            processed_image = processed_image.permute(1, 2, 0).numpy()

            if processed_image.dtype != np.uint8:
                processed_image = (processed_image * 255).astype(np.uint8)

             # Assuming 'processed_image' is a NumPy array in RGGB format
            processed_image = Image.fromarray(processed_image)

            # # Explicit image check
            # visualization_image = processed_dict['image'][:3]  # Extract RGB channels
            # visualization_image = visualization_image.cpu().detach().numpy()
            # visualization_image = visualization_image.transpose(1, 2, 0)
            # visualization_image = (visualization_image * 255).clip(0, 255).astype(np.uint8)
            # Image.fromarray(visualization_image).save("output_rgb.png")

            if self.transform:
                image = self.transform(processed_image)

            # visualization_image = image['global_crops'][0][:3]  # Extract RGB channels
            # visualization_image = visualization_image.cpu().detach().numpy()
            # visualization_image = visualization_image.transpose(1, 2, 0)
            # visualization_image = (visualization_image * 255).clip(0, 255).astype(np.uint8)
            # Image.fromarray(visualization_image).save("output_rgb_2.png")

            if self.target_transform and label is not None:
                label = self.target_transform(label)
            # processed_image.save("output.png")  

            return image, label
            
        except (EOFError, ValueError, Exception) as e:
            print(f"Error loading file {img_path}: {str(e)}")
            placeholder = torch.zeros((4, 128, 128))
            return placeholder, label

    
    def get_targets(self):
        """Returns a list of all labels in the dataset."""
        return [label for _, label in self.data]