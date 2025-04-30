import torch
import numpy as np
import os
import argparse
from torch.utils.data import DataLoader
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import json
from pathlib import Path
from PIL import Image
from torchvision import transforms

# Implementation of your UniformDataset class (for completeness)
class RAWDataPreProcessor:
    def __init__(self, mean=None, std=None, black_level=0, white_level=1.0):
        self.mean = mean
        self.std = std
        self.black_level = black_level
        self.white_level = white_level
        
    def __call__(self, data_dict):
        image = data_dict['image']
        
        # Normalize based on black and white levels
        if self.black_level != 0 or self.white_level != 1.0:
            image = (image - self.black_level) / (self.white_level - self.black_level)
            image = torch.clamp(image, 0, 1)
        
        # Apply mean and std normalization if provided
        if self.mean is not None and self.std is not None:
            mean = torch.tensor(self.mean).view(-1, 1, 1)
            std = torch.tensor(self.std).view(-1, 1, 1)
            image = (image - mean) / std
            
        data_dict['image'] = image
        return data_dict

class UniformDataset(torch.utils.data.Dataset):
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
                label = label_dict.get(img_name, -1)

                self.data.append((img_path, label))

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
            processed_image = processed_image.permute(1, 2, 0).numpy()
            
            if processed_image.dtype != np.uint8:
                processed_image = (processed_image * 255).astype(np.uint8)
            
            processed_image = Image.fromarray(processed_image)
            if self.transform:
                image = self.transform(processed_image)

            if self.target_transform and label is not None:
                label = self.target_transform(label)
                
            return image, label
            
        except (EOFError, ValueError, Exception) as e:
            print(f"Error loading file {img_path}: {str(e)}")
            # Return a placeholder instead of failing
            placeholder = torch.zeros((4, 128, 128))
            return placeholder, label

    def get_targets(self):
        """Returns a list of all labels in the dataset."""
        return [label for _, label in self.data]


def parse_args():
    parser = argparse.ArgumentParser(description='KNN validation for encoder')
    parser.add_argument('--checkpoint', type=str, default="/home/paperspace/Documents/nika_space/dinov2/main_data/model_0002999.rank_0.pth", help='Path to the encoder checkpoint')
    parser.add_argument('--dataset_root', type=str, default="/home/paperspace/Documents/nika_space/main_dataset/", help='Path to the dataset root directory')
    parser.add_argument('--output_dir', type=str, default="res_knn/", help='Directory to save cluster results')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for inference')
    parser.add_argument('--n_neighbors', type=int, default=5, help='Number of neighbors for KNN')
    parser.add_argument('--n_clusters', type=int, default=10, help='Number of clusters for KMeans')
    parser.add_argument('--img_size', type=int, default=224, help='Image size for model input')
    parser.add_argument('--black_level', type=float, default=0, help='Black level for normalization')
    parser.add_argument('--white_level', type=float, default=1.0, help='White level for normalization')
    parser.add_argument('--mean', type=float, nargs='+', default=None, help='Mean values for normalization')
    parser.add_argument('--std', type=float, nargs='+', default=None, help='Std values for normalization')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='Device to run inference on')
    return parser.parse_args()

def load_encoder(checkpoint_path, device):
    """
    Load the encoder model from checkpoint.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Try to figure out the model structure from the checkpoint
    if 'model' in checkpoint:
        model = checkpoint['model']
    elif 'encoder' in checkpoint:
        model = checkpoint['encoder']
    elif 'state_dict' in checkpoint:
        # If we just have state_dict, we'll need to know the model architecture
        # This might need to be modified based on your specific encoder architecture
        print("Warning: Checkpoint contains only state_dict. Using default encoder architecture.")
        from torchvision.models import resnet50
        model = resnet50(pretrained=False)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        # Try to directly load the checkpoint as a model
        try:
            model = checkpoint
        except:
            raise ValueError(f"Unable to load model from checkpoint. Unknown format: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else type(checkpoint)}")
    
    model = model.to(device)
    model.eval()
    return model

def load_dataset(dataset_root, batch_size, img_size, mean=None, std=None, black_level=0, white_level=1.0):
    """
    Load the dataset using your UniformDataset class.
    """
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    
    # Create dataset instance
    dataset = UniformDataset(
        root=dataset_root,
        transform=transform,
        mean=mean,
        std=std,
        black_level=black_level,
        white_level=white_level
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False
    )
    
    return dataloader, dataset.label_map

def extract_features(model, dataloader, device):
    """
    Extract features from the encoder for all samples in the dataset.
    """
    features = []
    labels = []
    paths = []  # Store paths for reference
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Extracting features")):
            if len(batch) >= 2:
                inputs, targets = batch[0].to(device), batch[1]
                
                # Skip samples with invalid labels (which have label -1)
                valid_indices = targets != -1
                if not valid_indices.any():
                    continue
                
                inputs = inputs[valid_indices]
                targets = targets[valid_indices]
                
                if len(inputs) == 0:
                    continue
                
                # Handle potential errors gracefully
                try:
                    # Get model output - might need adjusting based on your model
                    outputs = model(inputs)
                    
                    # If model returns a tuple or dict, extract the features
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]  # Assuming first element contains features
                    elif isinstance(outputs, dict) and 'features' in outputs:
                        outputs = outputs['features']
                    elif hasattr(model, 'fc') and isinstance(model.fc, torch.nn.Module):
                        # For models like ResNet that have a final fc layer, get the features before fc
                        # This is a common approach, but may need adjustment for your specific model
                        outputs = outputs  # Already extracted by the model
                    
                    features.append(outputs.cpu().numpy())
                    labels.append(targets.numpy())
                    
                except Exception as e:
                    print(f"Error processing batch {i}: {str(e)}")
                    continue
    
    if not features:
        raise ValueError("No features were extracted. Check your dataset and model.")
    
    features = np.vstack(features)
    labels = np.concatenate(labels)
    
    return features, labels

def evaluate_knn(features, labels, n_neighbors):
    """
    Evaluate the features using KNN classifier.
    """
    # Filter out samples with label -1 (unlabeled)
    valid_indices = labels != -1
    features_valid = features[valid_indices]
    labels_valid = labels[valid_indices]
    
    if len(np.unique(labels_valid)) < 2:
        print("Warning: Less than 2 unique labels found. KNN evaluation skipped.")
        return {
            'accuracy': None,
            'classification_report': None,
            'knn_model': None
        }
    
    # Split the data into train and test sets (80/20 split)
    n_samples = len(features_valid)
    indices = np.random.permutation(n_samples)
    train_size = int(0.8 * n_samples)
    
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    X_train, y_train = features_valid[train_indices], labels_valid[train_indices]
    X_test, y_test = features_valid[test_indices], labels_valid[test_indices]
    
    # Train KNN classifier
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    
    # Evaluate
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    return {
        'accuracy': accuracy,
        'classification_report': report,
        'knn_model': knn
    }

def perform_clustering(features, n_clusters, output_dir):
    """
    Perform KMeans clustering on the features and save results.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features)
    
    # Save cluster centers and labels
    cluster_centers = kmeans.cluster_centers_
    np.save(os.path.join(output_dir, 'cluster_centers.npy'), cluster_centers)
    np.save(os.path.join(output_dir, 'cluster_labels.npy'), cluster_labels)
    
    # Save KMeans model
    with open(os.path.join(output_dir, 'kmeans_model.pkl'), 'wb') as f:
        pickle.dump(kmeans, f)
    
    # Calculate inertia (sum of squared distances to closest cluster center)
    inertia = kmeans.inertia_
    
    # Calculate silhouette score if there are enough samples
    if len(features) > n_clusters:
        from sklearn.metrics import silhouette_score
        try:
            silhouette_avg = silhouette_score(features, cluster_labels)
        except:
            silhouette_avg = None
    else:
        silhouette_avg = None
    
    # Create a visualization of the clusters using t-SNE or PCA
    try:
        from sklearn.decomposition import PCA
        
        # Use PCA to reduce to 2D for visualization
        pca = PCA(n_components=2)
        reduced_features = pca.fit_transform(features)
        
        # Plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], 
                             c=cluster_labels, cmap='viridis', alpha=0.5)
        plt.colorbar(scatter, label='Cluster')
        plt.title('PCA visualization of clusters')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.savefig(os.path.join(output_dir, 'cluster_visualization.png'))
        
        # Optionally also do t-SNE (more computationally intensive)
        if len(features) < 10000:  # Only do t-SNE for smaller datasets
            from sklearn.manifold import TSNE
            tsne = TSNE(n_components=2, random_state=42)
            reduced_features_tsne = tsne.fit_transform(features)
            
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(reduced_features_tsne[:, 0], reduced_features_tsne[:, 1], 
                                c=cluster_labels, cmap='viridis', alpha=0.5)
            plt.colorbar(scatter, label='Cluster')
            plt.title('t-SNE visualization of clusters')
            plt.savefig(os.path.join(output_dir, 'tsne_visualization.png'))
    except Exception as e:
        print(f"Visualization failed: {e}")
    
    # Save cluster statistics
    cluster_counts = np.bincount(cluster_labels, minlength=n_clusters)
    cluster_stats = {
        'n_clusters': n_clusters,
        'inertia': float(inertia),
        'silhouette_score': float(silhouette_avg) if silhouette_avg is not None else None,
        'cluster_sizes': cluster_counts.tolist(),
    }
    
    with open(os.path.join(output_dir, 'cluster_stats.json'), 'w') as f:
        json.dump(cluster_stats, f, indent=2)
    
    return {
        'cluster_labels': cluster_labels,
        'kmeans': kmeans,
        'stats': cluster_stats
    }

def analyze_clusters(features, cluster_labels, label_map, output_dir):
    """
    Analyze the clusters to see if they correspond to meaningful categories.
    """
    inverse_label_map = {v: k for k, v in label_map.items()}
    
    # Count instances of each label in each cluster
    n_clusters = max(cluster_labels) + 1
    label_distribution = {}
    
    unique_labels = np.unique(labels)
    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_features = features[cluster_mask]
        cluster_labels_subset = labels[cluster_mask]
        
        # Count occurrences of each label in this cluster
        label_counts = {}
        for label in unique_labels:
            if label == -1:  # Skip unlabeled data
                continue
            count = np.sum(cluster_labels_subset == label)
            if count > 0:
                label_text = inverse_label_map.get(label, f"Unknown-{label}")
                label_counts[label_text] = int(count)
        
        label_distribution[f"cluster_{cluster_id}"] = label_counts
    
    # Save the analysis
    with open(os.path.join(output_dir, 'cluster_analysis.json'), 'w') as f:
        json.dump(label_distribution, f, indent=2)
    
    return label_distribution

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Log the configuration
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Load model
    print(f"Loading encoder from {args.checkpoint}")
    model = load_encoder(args.checkpoint, args.device)
    
    # Load dataset
    print(f"Loading dataset from {args.dataset_root}")
    dataloader, label_map = load_dataset(
        args.dataset_root, 
        args.batch_size,
        args.img_size,
        mean=args.mean,
        std=args.std,
        black_level=args.black_level,
        white_level=args.white_level
    )
    
    # Save label mapping
    with open(os.path.join(args.output_dir, 'label_map.json'), 'w') as f:
        # Convert numeric keys to strings for JSON
        serializable_map = {str(k): int(v) for k, v in label_map.items()}
        json.dump(serializable_map, f, indent=2)
    
    # Extract features
    print("Extracting features...")
    features, labels = extract_features(model, dataloader, args.device)
    
    # Save features
    print(f"Saving features to {args.output_dir}")
    np.save(os.path.join(args.output_dir, 'features.npy'), features)
    np.save(os.path.join(args.output_dir, 'labels.npy'), labels)
    
    # Evaluate with KNN if we have labels
    if len(np.unique(labels)) > 1 and np.sum(labels != -1) > 0:
        print(f"Evaluating with KNN (k={args.n_neighbors})...")
        knn_results = evaluate_knn(features, labels, args.n_neighbors)
        
        if knn_results['accuracy'] is not None:
            print(f"KNN Accuracy: {knn_results['accuracy']:.4f}")
            
            # Save KNN results
            with open(os.path.join(args.output_dir, 'knn_results.json'), 'w') as f:
                # Convert numpy types to Python native types for JSON serialization
                results_json = {
                    'accuracy': float(knn_results['accuracy']),
                    'classification_report': {
                        k: {
                            kk: float(vv) for kk, vv in v.items()
                        } if isinstance(v, dict) else v
                        for k, v in knn_results['classification_report'].items()
                    }
                }
                json.dump(results_json, f, indent=2)
            
            # Save KNN model
            with open(os.path.join(args.output_dir, 'knn_model.pkl'), 'wb') as f:
                pickle.dump(knn_results['knn_model'], f)
    
    # Perform clustering
    print(f"Performing KMeans clustering (k={args.n_clusters})...")
    clustering_results = perform_clustering(features, args.n_clusters, 
                                          os.path.join(args.output_dir, 'clusters'))
    
    # Analyze clusters
    if len(label_map) > 0:
        print("Analyzing cluster composition...")
        label_distribution = analyze_clusters(
            features, 
            clustering_results['cluster_labels'], 
            label_map, 
            os.path.join(args.output_dir, 'clusters')
        )
    
    print(f"Clustering complete. Results saved to {args.output_dir}")
    
    # Save summary info in a README
    summary = f"""
# Encoder Validation Results

## Configuration
- Checkpoint: {args.checkpoint}
- Dataset: {args.dataset_root}
- Number of samples: {len(features)}
- Feature dimension: {features.shape[1]}
- KNN neighbors: {args.n_neighbors}
- Number of clusters: {args.n_clusters}

## Results Summary
"""
    
    if 'knn_results' in locals() and knn_results['accuracy'] is not None:
        summary += f"""
### KNN Classification
- Accuracy: {knn_results['accuracy']:.4f}
- See knn_results.json for detailed metrics
"""
    
    summary += f"""
### Clustering
- Number of clusters: {args.n_clusters}
- Inertia: {clustering_results['stats']['inertia']:.2f}
"""
    
    if clustering_results['stats']['silhouette_score'] is not None:
        summary += f"- Silhouette score: {clustering_results['stats']['silhouette_score']:.4f}\n"
    
    summary += """
## Files
- features.npy: Extracted features
- labels.npy: Original labels
- label_map.json: Mapping between text labels and numeric indices
- knn_results.json: KNN classification results (if applicable)
- knn_model.pkl: Trained KNN model (if applicable)
- clusters/: Clustering results directory
  - cluster_centers.npy: Cluster centers
  - cluster_labels.npy: Cluster assignments
  - cluster_stats.json: Clustering statistics
  - cluster_analysis.json: Analysis of label distribution in clusters
  - kmeans_model.pkl: Trained KMeans model
  - cluster_visualization.png: PCA visualization of clusters
  - tsne_visualization.png: t-SNE visualization (if available)
"""
    
    with open(os.path.join(args.output_dir, 'README.md'), 'w') as f:
        f.write(summary)

if __name__ == "__main__":
    main()