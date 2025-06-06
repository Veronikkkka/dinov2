import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import faiss
import numpy as np
import shutil
from tqdm import tqdm
from torchvision.utils import save_image
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F


DATASET_PATH = "/home/paperspace/Documents/nika_space/ADE20K/ADEChallengeData2016/images/training_raw/" 
CHECKPOINT_PATH = "/home/paperspace/Documents/nika_space/dinov2/lala3/model_0010499.rank_0.pth"  
SAVE_DIR = "/home/paperspace/Documents/nika_space/dinov2/lala3/res1/"
KNN_K = 5 
NUM_EXAMPLES = 10 
IMAGE_SIZE = 224
from dinov2.train.ssl_meta_arch import get_downloaded_dino_vit_interpolated

import torch
import torch.nn as nn

def load_dino_model(checkpoint_path, arch="dinov2_vitb14", merge_block_indexes=None):
    student_backbone = get_downloaded_dino_vit_interpolated(arch, merge_block_indexes)
    
    pre_encoder = getattr(student_backbone, "pre_encoder", nn.Identity())
    merge_blocks = getattr(student_backbone, "merge_blocks", nn.Identity())
    model_adapter = getattr(student_backbone, "model_adapter", nn.Identity())

    model = nn.ModuleDict({
        "backbone": student_backbone,
        "pre_encoder": pre_encoder,
        "merge_blocks": merge_blocks,
        "model_adapter": model_adapter
    })

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    print("State dict keys: ", state_dict.keys())

    adjusted_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("backbone."):
            adjusted_key = key[len("backbone."):]
            adjusted_state_dict[f"backbone.{adjusted_key}"] = value
        else:
            adjusted_state_dict[key] = value

    model.load_state_dict(adjusted_state_dict, strict=False)
    model.eval()

    return model

from dinov2.data.datasets import ADK20Dataset
def get_data_loader(dataset_path, image_size, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    dataset = ADK20Dataset(dataset_path, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return loader, dataset


def extract_features(model, dataset, batch_size=32, num_workers=4, device="cuda"):
    """
    Extracts features from the dataset using the provided DINO model.

    Args:
        model (torch.nn.Module): Pretrained DINO model.
        dataset (Dataset): ADK20Dataset instance.
        batch_size (int): Batch size for DataLoader.
        num_workers (int): Number of workers for DataLoader.
        device (str): Device to run inference on ("cuda" or "cpu").

    Returns:
        features (torch.Tensor): Extracted feature vectors.
        labels (torch.Tensor): Corresponding labels.
        filepaths (list): Filepaths of images.
    """
    model.to(device)
    model.eval()

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    features_list = []
    labels_list = []
    filepaths_list = []

    with torch.no_grad():
        for images, targets, filepaths in tqdm(dataloader, desc="Extracting Features"):
            images = images.to(device)
            features = model["backbone"](images)
            features = F.normalize(features, dim=1)
            
            features_list.append(features.cpu())
            labels_list.append(targets.cpu())
            filepaths_list.extend(filepaths)

    features = torch.cat(features_list, dim=0)
    labels = torch.cat(labels_list, dim=0)

    return features, labels, filepaths_list


def knn_classification(train_features, train_labels, test_features, k=5):
    """
    Performs k-NN classification.

    Args:
        train_features (torch.Tensor): Training feature vectors.
        train_labels (torch.Tensor): Training labels.
        test_features (torch.Tensor): Test feature vectors.
        k (int): Number of nearest neighbors.

    Returns:
        predictions (torch.Tensor): Predicted labels for test set.
    """
    similarities = test_features @ train_features.T 
    top_k_indices = similarities.topk(k, dim=1).indices 

    top_k_labels = train_labels[top_k_indices]

   
    predictions = torch.mode(top_k_labels, dim=1).values

    return predictions

def cluster_with_knn(features, k=KNN_K):
    index = faiss.IndexFlatL2(features.shape[1])
    index.add(features)
    _, neighbors = index.search(features, k + 1)
    return neighbors[:, 1:] 

def save_clustered_images(dataset, predictions, save_dir, num_examples=10):
    """
    Saves images grouped into clusters (based on predictions).

    Args:
        dataset (ADK20Dataset): The dataset object, which contains image paths.
        predictions (torch.Tensor or np.ndarray): Predicted cluster IDs for each image.
        save_dir (str): Directory where clustered images will be saved.
        num_examples (int): Number of examples to save per cluster.
    """
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    image_paths = dataset.image_paths

    unique_predictions = torch.unique(predictions) if isinstance(predictions, torch.Tensor) else np.unique(predictions)

    for cluster_id in unique_predictions:
        cluster_path = os.path.join(save_dir, f"cluster_{cluster_id.item()}")
        os.makedirs(cluster_path, exist_ok=True)

        cluster_indices = torch.where(predictions == cluster_id)[0].tolist()

        for i, idx in enumerate(cluster_indices[:num_examples]):
            img_path = str(image_paths[idx])
            try:
                img = Image.open(img_path).convert("RGB")
                img.save(os.path.join(cluster_path, f"img_{i}.jpg"))
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")

    print(f"Clustered images saved to {save_dir}")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_dino_model(CHECKPOINT_PATH)
    train_dataset = ADK20Dataset(DATASET_PATH, transform=transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ]))
    test_dataset = ADK20Dataset(DATASET_PATH, transform=transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ]))
    print("Extracting features...")
    train_features, train_labels, _ = extract_features(model, train_dataset)
    test_features, _, _ = extract_features(model, test_dataset)

    print("Performing KNN clustering...")
    predictions = knn_classification(train_features, train_labels, test_features, k=5)

    print("Saving cluster examples...")
    save_clustered_images(test_dataset, predictions, save_dir=SAVE_DIR, num_examples=5)
    print(f"Clustered images saved in {SAVE_DIR}")

if __name__ == "__main__":
    main()