import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

class UniformDataset(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        """
        Args:
            root (str): Path to the main dataset directory.
            transform (callable, optional): Transformations to apply to the images.
            target_transform (callable, optional): Transformations to apply to labels.
        """
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
                            continue

                        filename, *label_text = parts
                        label_text = " ".join(label_text)

                        if label_text not in self.label_map:
                            self.label_map[label_text] = label_idx
                            label_idx += 1

                        label_dict[filename] = self.label_map[label_text]

            for img_name in sorted(os.listdir(images_folder)):
                img_path = os.path.join(images_folder, img_name)
                label = label_dict.get(img_name, -1)  # -1 if no label

                self.data.append((img_path, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]


        # image = Image.open(img_path).convert("RGB")
        print("Img path: ", img_path)
        image = np.load(img_path)
        image = torch.from_numpy(image).float()

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def get_targets(self):
        """Returns a list of all labels in the dataset."""
        return [label for _, label in self.data]
