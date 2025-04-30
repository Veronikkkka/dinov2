import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import os
from PIL import Image
import numpy as np
import sys
from dinov2.eval.linear import get_args_parser
from dinov2.data.transforms import make_classification_eval_transform, make_classification_train_transform
from dinov2.data.datasets.pre_processor import RAWDataPreProcessor
from dinov2.eval.setup import setup_and_build_model
import random

class DINOv2SegmentationModel(nn.Module):
    def __init__(self, dinov2_backbone, num_classes=150):
        """
        Segmentation model using DINOv2 as a backbone
        
        Args:
            dinov2_backbone: Pre-trained DINOv2 model loaded with build_model_for_eval
            num_classes: Number of segmentation classes (150 for ADE20K)
        """
        super().__init__()
        self.backbone = dinov2_backbone
        
        self.feature_dim = self.backbone.embed_dim
        print(f"Using feature dimension: {self.feature_dim}")
        
        
        self.params_to_update = []
        for name, param in self.backbone.named_parameters():
            param.requires_grad = True
            self.params_to_update.append(param)
        
        self.decoder = SegmentationHead(self.feature_dim, num_classes)
        
    def forward(self, x):

        features = self.backbone.get_intermediate_layers(x, n=1)[0]  # [B, N+1, C]

        patch_tokens = features[:, 1:, :]  # [B, 255, C]

        batch_size = x.shape[0]

        patch_tokens = patch_tokens.transpose(1, 2)  # [B, C, 255]
        patch_tokens = patch_tokens.reshape(batch_size, patch_tokens.shape[1], 15, 17)  # [B, C, 15, 17]

        patch_tokens = F.interpolate(patch_tokens, size=(16, 16), mode="bilinear", align_corners=False)


        upsampled_features = F.interpolate(
            patch_tokens,
            size=(x.shape[2], x.shape[3]), 
            mode="bilinear",
            align_corners=False
        )

        segmentation_map = self.decoder(upsampled_features)

        return segmentation_map



class SegmentationHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 512, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.dropout = nn.Dropout(0.1)
        self.conv4 = nn.Conv2d(128, num_classes, kernel_size=1)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        x = self.conv4(x)
        return x


class CustomSegmentationDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, target_transform=None):
        """
        Custom segmentation dataset for raw images with PNG masks
        
        Args:
            root_dir: Path to dataset root
            split: 'train', 'val', or 'test'
            transform: Transformations for input images
            target_transform: Transformations for masks
        """
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        
        self.data_dir = os.path.join(root_dir, split, 'ade')
        
        self.images_dir = os.path.join(self.data_dir, 'images') if os.path.exists(os.path.join(self.data_dir, 'images')) else self.data_dir
        
        if split == "train":
            self.masks_dir = "/home/paperspace/Documents/nika_space/ADE20K/ADEChallengeData2016/annotations/training/"
        else:
            self.masks_dir = "/home/paperspace/Documents/nika_space/ADE20K/ADEChallengeData2016/annotations/validation/"
        
        self.images = [f for f in os.listdir(self.images_dir) if f.endswith(".npy")]
        
        print(f"Found {len(self.images)} images in {self.data_dir}")
        
        self.preprocessor = RAWDataPreProcessor(
            mean=None,
            std=None,
            black_level=0,
            white_level=1,
        )
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):

        img_name = self.images[idx]
        img_path = os.path.join(self.images_dir, img_name)
        
        if self.split == "train":
            mask_name = img_name.replace(".npy", ".png")
            mask_path = os.path.join(self.masks_dir, mask_name)
        else:
            mask_name = img_name.replace("_train_", "_val_").replace(".npy", ".png")
            mask_path = os.path.join(self.masks_dir, mask_name)
        

        if not os.path.exists(mask_path):
            alt_mask_name = img_name.replace(".npy", ".png")
            mask_path = os.path.join(self.masks_dir.replace("validation", "training"), alt_mask_name)
            
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Mask not found for {img_name} at {mask_path}")
        
        image = np.load(img_path, allow_pickle=True)
        
        if image.dtype == np.uint16:
            image = image.astype(np.float32) 
            

        if len(image.shape) == 3 and image.shape[0] == 4:
            raw_tensor = torch.from_numpy(image).float()
            processed_dict = self.preprocessor({"image": raw_tensor})
            processed_image = processed_dict['image']
            
            if self.transform:
                image = self.transform(processed_image)
            else:
                image = processed_image
        else:
            raise ValueError(f"Unexpected image format: {image.shape}")
        
        mask = Image.open(mask_path)
        mask = np.array(mask)
        mask_tensor = torch.from_numpy(mask).long()
        
        if self.target_transform:
            mask = self.target_transform(mask_tensor)
        else:
            mask = mask_tensor
        
        if image.shape[-2:] != mask.shape[-2:]:
            mask = F.interpolate(
                mask.unsqueeze(0).float(), 
                size=(image.shape[-2], image.shape[-1]),
                mode='nearest'
            ).squeeze(0).long()
        
        return image, mask


def count_dataset_classes(dataset):
    class_set = set()
    
    sample_size = min(100, len(dataset))
    indices = list(range(len(dataset)))
    
    random_indices = random.sample(indices, sample_size) if len(dataset) > sample_size else indices
    
    for i in random_indices:
        try:
            _, mask = dataset[i]
            unique_classes = torch.unique(mask)
            for cls in unique_classes:
                class_set.add(cls.item())
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue
    
    num_classes = len(class_set)
    max_class = max(class_set) if class_set else 0
    
    print(f"Detected classes: {sorted(list(class_set))}")
    print(f"Number of unique classes: {num_classes}")
    print(f"Maximum class ID: {max_class}")
    
    return max_class + 1


import time

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cuda'):

    model.to(device)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
    
    best_miou = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        epoch_start_time = time.time()
        
        for i, (images, masks) in enumerate(train_loader):
            batch_start_time = time.time()
            
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            
            batch_time = time.time() - batch_start_time
            
            batches_left = len(train_loader) - i - 1
            epoch_time_left = batches_left * batch_time
            
            if i % 10 == 0:
                eta = time.strftime('%H:%M:%S', time.gmtime(epoch_time_left))
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {i}/{len(train_loader)}, Loss: {loss.item():.4f}, ETA: {eta}')
        
        train_loss = train_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        val_miou = 0.0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                val_loss += loss.item() * images.size(0)
                
  
                preds = torch.argmax(outputs, dim=1)
                val_miou += calculate_miou(preds, masks, num_classes=model.decoder.conv4.out_channels).item()
        
        val_loss = val_loss / len(val_loader.dataset)
        val_miou = val_miou / len(val_loader)
        

        scheduler.step(val_loss)
        
        epoch_time = time.time() - epoch_start_time
        eta_epoch = time.strftime('%H:%M:%S', time.gmtime((num_epochs - epoch - 1) * epoch_time))
        
        print(f'Epoch {epoch+1}/{num_epochs}, '
              f'Train Loss: {train_loss:.4f}, '
              f'Val Loss: {val_loss:.4f}, '
              f'Val mIoU: {val_miou:.4f}, '
              f'ETA for completion: {eta_epoch}')
        
        
        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(model.state_dict(), 'best_dinov2_segmentation.pth')
            print(f"Saved best model with mIoU: {best_miou:.4f}")
    
    return model



def calculate_miou(preds, targets, num_classes):
    """
    Calculate mean IoU (Intersection over Union) for segmentation evaluation
    
    Args:
        preds: Predicted segmentation maps
        targets: Ground truth segmentation maps
        num_classes: Number of segmentation classes
    """
    ious = []
    smooth = 1e-6
    
    for cls in range(num_classes):
        pred_mask = (preds == cls)
        target_mask = (targets == cls)
        
        if target_mask.sum() == 0 and pred_mask.sum() == 0:
            continue
            
        intersection = torch.logical_and(pred_mask, target_mask).sum()
        union = torch.logical_or(pred_mask, target_mask).sum()
        
        iou = (intersection + smooth) / (union + smooth)
        ious.append(iou)
    
    return torch.mean(torch.stack(ious)) if ious else torch.tensor(0.0)


def test_model(model, test_loader, device='cuda'):
    """
    Test the trained model and calculate mIoU
    
    Args:
        model: Trained segmentation model
        test_loader: DataLoader for test data
        device: Device to test on
        
    Returns:
        Mean IoU on test set
    """
    model.to(device)
    model.eval()
    
    test_miou = 0.0
    
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            test_miou += calculate_miou(preds, masks, num_classes=model.decoder.conv4.out_channels).item()
    
    test_miou = test_miou / len(test_loader)
    print(f"Test mIoU: {test_miou:.4f}")
    
    return test_miou


def main(args):
    model, autocast_dtype = setup_and_build_model(args)
    
    train_transform = make_classification_train_transform(
        hflip_prob=0.5,  
        crop_size=224,
    )
    
    val_transform = make_classification_eval_transform(
        crop_size=224 
    )
    

    mask_train_transform = transforms.Compose([
        transforms.Lambda(lambda x: x.unsqueeze(0) if x.dim() == 2 else x), 
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.Lambda(lambda x: x.squeeze(0) if x.shape[0] == 1 else x) 
    ])
    
    mask_val_transform = transforms.Compose([
        transforms.Lambda(lambda x: x.unsqueeze(0) if x.dim() == 2 else x), 
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.Lambda(lambda x: x.squeeze(0) if x.shape[0] == 1 else x)
    ])
    

    dataset_root = "/home/paperspace/Documents/nika_space/main_dataset/" 

    train_dataset = CustomSegmentationDataset(
        dataset_root, 
        split='train',
        transform=train_transform,
        target_transform=mask_train_transform
    )
    
    val_dataset = CustomSegmentationDataset(
        dataset_root, 
        split='val',
        transform=val_transform,
        target_transform=mask_val_transform
    )
    
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    try:
        sample_img, sample_mask = train_dataset[0]
        print(f"Sample image shape: {sample_img.shape}")
        print(f"Sample mask shape: {sample_mask.shape}")
        print(f"Sample mask unique values: {torch.unique(sample_mask)}")
    except Exception as e:
        print(f"Error loading sample: {e}")
    

    num_classes = count_dataset_classes(train_dataset)
    print(f"Using {num_classes} classes for segmentation")
    

    segmentation_model = DINOv2SegmentationModel(model, num_classes=num_classes)
    

    train_loader = DataLoader(
        train_dataset, 
        batch_size=4, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=4, 
        num_workers=4,
        pin_memory=True
    )
    
    criterion = nn.CrossEntropyLoss()
    
    params = [
        {'params': segmentation_model.params_to_update, 'lr': 1e-5},  # backbone fine-tuning
        {'params': segmentation_model.decoder.parameters(), 'lr': 1e-4}  # decoder
    ]
    
    optimizer = optim.AdamW(params, weight_decay=1e-5)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    trained_model = train_model(
        segmentation_model, 
        train_loader, 
        val_loader, 
        criterion, 
        optimizer,
        num_epochs=30,
        device=device
    )
    
    torch.save(trained_model.state_dict(), 'final_dinov2_segmentation.pth')
    print("Training complete. Final model saved.")


if __name__ == "__main__":
    args_parser = get_args_parser()
    args = args_parser.parse_args()
    sys.exit(main(args))