import argparse
from functools import partial
import json
import logging
import os
import sys
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from fvcore.common.checkpoint import Checkpointer, PeriodicCheckpointer

from dinov2.data import SamplerType, make_data_loader, make_dataset
from dinov2.data.transforms import make_classification_eval_transform
import dinov2.distributed as distributed
from dinov2.eval.metrics import MetricType, build_metric
from dinov2.eval.setup import get_args_parser as get_setup_args_parser
from dinov2.eval.setup import setup_and_build_model
from dinov2.eval.utils import ModelWithIntermediateLayers, evaluate
from dinov2.logging import MetricLogger

logger = logging.getLogger("dinov2_segmentation")

import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image
import os
from torchvision.utils import make_grid

def visualize_segmentation_predictions(model, data_loader, device, output_dir, num_samples=5):
    """
    Visualize segmentation predictions compared to ground truth
    
    Args:
        model: The segmentation model
        data_loader: DataLoader for validation/test data
        device: Device to run inference on
        output_dir: Directory to save visualizations
        num_samples: Number of samples to visualize
    """
    # Create output directory if it doesn't exist
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Set model to eval mode
    model.eval()
    
    # Color map for segmentation mask visualization (adjust for your num_classes)
    # This is a simple color map, you can create a more sophisticated one based on dataset
    colormap = torch.randint(0, 256, (150, 3), dtype=torch.uint8)
    colormap[0] = torch.tensor([0, 0, 0], dtype=torch.uint8)  # Background usually black
    
    with torch.no_grad():
        for i, (images, targets) in enumerate(data_loader):
            if i >= num_samples:
                break
                
            # Move to device
            images = images.to(device)
            targets = targets.to(device)
            
            # Get predictions
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            
            # Process each image in batch
            for j in range(images.shape[0]):
                # Skip if we've already processed enough samples
                if i*data_loader.batch_size + j >= num_samples:
                    break
                    
                # Get single image and its prediction
                image = images[j].cpu()
                target = targets[j].cpu()
                pred = preds[j].cpu()
                
                # Create figure
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # Plot input image (adjust for RGGB visualization)
                if image.shape[0] == 4:  # RGGB format
                    # Simplified RGB conversion for visualization
                    # R = image[0], G = (image[1] + image[2])/2, B = image[3]
                    rgb_image = torch.stack([
                        image[0],
                        (image[1] + image[2]) / 2,
                        image[3]
                    ], dim=0)
                    
                    # Normalize for display
                    rgb_min = rgb_image.min()
                    rgb_max = rgb_image.max()
                    rgb_image = (rgb_image - rgb_min) / (rgb_max - rgb_min)
                    
                    axes[0].imshow(rgb_image.permute(1, 2, 0).numpy())
                else:
                    # Standard RGB/grayscale image
                    axes[0].imshow(image.permute(1, 2, 0).numpy())
                axes[0].set_title("Input Image")
                axes[0].axis('off')
                
                # Plot ground truth mask
                target_vis = colormap[target].permute(2, 0, 1)
                axes[1].imshow(target_vis.permute(1, 2, 0).numpy())
                axes[1].set_title("Ground Truth")
                axes[1].axis('off')
                
                # Plot prediction mask
                pred_vis = colormap[pred].permute(2, 0, 1)
                axes[2].imshow(pred_vis.permute(1, 2, 0).numpy())
                axes[2].set_title("Prediction")
                axes[2].axis('off')
                
                # Save figure
                sample_idx = i*data_loader.batch_size + j
                plt.tight_layout()
                plt.savefig(os.path.join(vis_dir, f"segmentation_sample_{sample_idx}.png"))
                plt.close(fig)
                
                # Also save a visualization of class distribution
                plt.figure(figsize=(10, 5))
                unique_target, counts_target = torch.unique(target, return_counts=True)
                unique_pred, counts_pred = torch.unique(pred, return_counts=True)
                
                plt.subplot(1, 2, 1)
                plt.bar([str(x.item()) for x in unique_target], counts_target.numpy())
                plt.title("Ground Truth Class Distribution")
                plt.xlabel("Class ID")
                plt.ylabel("Pixel Count")
                
                plt.subplot(1, 2, 2)
                plt.bar([str(x.item()) for x in unique_pred], counts_pred.numpy())
                plt.title("Predicted Class Distribution")
                plt.xlabel("Class ID")
                plt.ylabel("Pixel Count")
                
                plt.tight_layout()
                plt.savefig(os.path.join(vis_dir, f"class_dist_sample_{sample_idx}.png"))
                plt.close()
    
    print(f"Saved visualizations to {vis_dir}")


class DINOv2SegmentationDecoder(nn.Module):
    """Decoder module for segmentation using DINOv2 features"""
    def __init__(self, dinov2_dim=768, hidden_dim=224, num_classes=21):
        super().__init__()
        
        self.project = nn.Conv2d(dinov2_dim, hidden_dim, kernel_size=1)
        
 
        self.upsample1 = UpsampleBlock(hidden_dim, hidden_dim // 2)
        self.upsample2 = UpsampleBlock(hidden_dim // 2, hidden_dim // 4)
        self.upsample3 = UpsampleBlock(hidden_dim // 4, hidden_dim // 8)
        

        self.classifier = nn.Conv2d(hidden_dim // 8, num_classes, kernel_size=1)

    def forward(self, features):

        x = self.project(features)

        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.upsample3(x)

        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)  # Resize directly to 224x224
        
        x = self.classifier(x)
        
        return x


class DINOv2SegmentationModel(nn.Module):
    """Full segmentation model with DINOv2 encoder and custom decoder"""
    def __init__(self, dinov2_model, num_classes=21, patch_size=14, img_size=224):
        super().__init__()
        self.encoder = dinov2_model
        
        self.dinov2_dim = dinov2_model.embed_dim
        self.patch_size = patch_size
        self.img_size = img_size
        
        self.encoder.eval()
        

        self.decoder = DINOv2SegmentationDecoder(
            dinov2_dim=self.dinov2_dim,
            hidden_dim=256,
            num_classes=num_classes
        )

    def extract_features(self, x):
        """Extract patch features from DINOv2 encoder"""

        with torch.no_grad():

            patch_tokens = self.encoder.get_intermediate_layers(x, n=1)[0][0]
            
            batch_size = x.shape[0]
            num_tokens = patch_tokens.shape[0]
            token_dim = patch_tokens.shape[1]

            num_patches_per_image = num_tokens // batch_size
            h = w = int(np.sqrt(num_patches_per_image))

            assert h * w == num_patches_per_image, "Patch tokens can't form square grid."

            patch_tokens = patch_tokens.reshape(batch_size, h, w, token_dim).permute(0, 3, 1, 2)

        return patch_tokens

    def forward(self, x):
        """Forward pass through encoder and decoder"""

        features = self.extract_features(x)
        

        logits = self.decoder(features)
        
        return logits

    def prepare_for_distributed_training(self):
        """Prepare model for distributed training using FSDP"""
        logger.info("DISTRIBUTED FSDP -- preparing segmentation model for distributed training")
        

        device_id = distributed.get_local_rank()
 
        fsdp_config = {
            "sharding_strategy": self.cfg.compute_precision.student.backbone.sharding_strategy,
            "mixed_precision": self.cfg.compute_precision.student.backbone.mixed_precision,
            "device_id": device_id,
            "sync_module_states": True,
            "use_orig_params": True,
        }
        

        self.decoder.project = FSDP(
            self.decoder.project,
            **fsdp_config
        )
        

        self.decoder.upsample1 = FSDP(
            self.decoder.upsample1,
            **fsdp_config
        )
        
        self.decoder.upsample2 = FSDP(
            self.decoder.upsample2,
            **fsdp_config
        )
        
        self.decoder.upsample3 = FSDP(
            self.decoder.upsample3,
            **fsdp_config
        )
        

        self.decoder.classifier = FSDP(
            self.decoder.classifier,
            **fsdp_config
        )
        
        logger.info("FSDP wrapping of segmentation decoder completed")


class SegmentationLoss(nn.Module):
    """Combined loss function for segmentation"""
    def __init__(self, weight=None):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight=weight, ignore_index=255)
        self.dice_loss = DiceLoss(ignore_index=255)
        
    def forward(self, preds, targets):
 
        ce = self.ce_loss(preds, targets)
        dice = self.dice_loss(preds, targets)
        return ce + dice


class DiceLoss(nn.Module):
    """Dice loss for segmentation"""
    def __init__(self, ignore_index=255, smooth=1.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth
        
    def forward(self, preds, targets):

        targets_one_hot = F.one_hot(
            targets.clamp(0), num_classes=preds.shape[1]
        ).permute(0, 3, 1, 2).float()
        

        mask = (targets != self.ignore_index).float().unsqueeze(1)
        

        preds_softmax = F.softmax(preds, dim=1)

        preds_softmax = preds_softmax * mask
        targets_one_hot = targets_one_hot * mask

        intersection = torch.sum(preds_softmax * targets_one_hot, dim=(0, 2, 3))
        union = torch.sum(preds_softmax, dim=(0, 2, 3)) + torch.sum(targets_one_hot, dim=(0, 2, 3))

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        dice_loss = 1.0 - dice.mean()
        
        return dice_loss


def get_fsdp_wrapper(cfg, modules_to_wrap=None):
    """Helper function to create an FSDP wrapper"""
    def wrapper(module):
        if modules_to_wrap is not None and not isinstance(module, tuple(modules_to_wrap)):
            return module
        return FSDP(
            module,
            sharding_strategy=cfg.sharding_strategy,
            mixed_precision=cfg.mixed_precision,
            device_id=distributed.get_local_rank(),
            sync_module_states=True,
            use_orig_params=True,
        )
    return wrapper


def get_args_parser(
    description: Optional[str] = None,
    parents: Optional[List[argparse.ArgumentParser]] = None,
    add_help: bool = True,
):
    parents = parents or []
    setup_args_parser = get_setup_args_parser(parents=parents, add_help=False)
    parents = [setup_args_parser]
    parser = argparse.ArgumentParser(
        description=description,
        parents=parents,
        add_help=add_help,
    )
    parser.add_argument(
        "--train-dataset",
        dest="train_dataset_str",
        type=str,
        help="Training dataset",
        default="Seg:root=/home/paperspace/Documents/nika_space/ADE20K/ADEChallengeData2016:split=train"
    )
    parser.add_argument(
        "--val-dataset",
        dest="val_dataset_str",
        type=str,
        help="Validation dataset",
        default="Seg:root=/home/paperspace/Documents/nika_space/ADE20K/ADEChallengeData2016:split=val"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch Size (per GPU)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        help="Number of Workers",
    )
    parser.add_argument(
        "--lr",
        type=float,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        help="Weight decay",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        help="Number of segmentation classes",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        help="DINOv2 patch size",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        help="Input image size",
    )
    parser.add_argument(
        "--eval-period-iterations",
        type=int,
        help="Number of iterations between evaluations",
    )
    parser.add_argument(
        "--save-checkpoint-frequency",
        type=int,
        help="Number of epochs between checkpoint saves",
    )
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Freeze DINOv2 encoder weights",
    )
    parser.add_argument(
        "--use-fsdp",
        action="store_true",
        help="Use Fully Sharded Data Parallel for distributed training",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=0,
        help="Local rank for distributed training",
    )
    parser.add_argument(
        "--rggb-mean",
        type=str,
        default="0.485,0.456,0.406,0.485",
        help="Mean values for RGGB channels normalization",
    )
    parser.add_argument(
        "--rggb-std",
        type=str,
        default="0.229,0.224,0.225,0.229",
        help="Std values for RGGB channels normalization",
    )
    parser.add_argument(
        "--black-level",
        type=float,
        default=0,
        help="Black level for raw images",
    )
    parser.add_argument(
        "--white-level",
        type=float,
        default=1.0,
        help="White level for raw images",
    )
    
    parser.set_defaults(
        train_dataset_str="Seg:root=/home/paperspace/Documents/nika_space/ADE20K/ADEChallengeData2016:split=train",
        val_dataset_str="Seg:root=/home/paperspace/Documents/nika_space/ADE20K/ADEChallengeData2016:split=val",
        epochs=50,
        batch_size=16,
        num_workers=8,
        lr=0.0001,
        weight_decay=0.0001,
        num_classes=21,
        patch_size=14,
        img_size=224,
        eval_period_iterations=500,
        save_checkpoint_frequency=5,
        freeze_encoder=True,
        use_fsdp=False,
    )
    return parser

from dinov2.data.datasets.pre_processor import RAWDataPreProcessor



def make_segmentation_transform(split="train", img_size=224):
    """Create segmentation data transforms for RGGB data"""
    from torchvision import transforms
    
    if split == "train":
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),

        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            
        ])


def make_segmentation_dataset(dataset_str, transform=None, split="train"):
    """Create segmentation dataset from RGGB .npy images and RGB PNG masks"""
    import os
    import torch
    from torch.utils.data import Dataset
    from PIL import Image
    import numpy as np
    from torchvision import transforms
    
    dataset_name, *params = dataset_str.split(":")
    params_dict = {}
    for param in params:
        if "=" in param:
            key, value = param.split("=")
            params_dict[key] = value

    class RGGBSegmentationDataset(Dataset):
        def __init__(self, root, split, transform=None, target_transform=None, mean=None, std=None, black_level=0, white_level=1.0):
            self.root = root
            self.split = split
            self.transform = transform
            self.target_transform = target_transform
            
            if split == "train":
                self.images_dir = "/home/paperspace/Documents/nika_space/main_dataset/train/ade/images"
                self.masks_dir = "/home/paperspace/Documents/nika_space/ADE20K/ADEChallengeData2016/annotations/training"
            else:
                self.images_dir = "/home/paperspace/Documents/nika_space/main_dataset/val/ade/images"
                self.masks_dir = "/home/paperspace/Documents/nika_space/ADE20K/ADEChallengeData2016/annotations/validation"


            self.image_paths = sorted([
                os.path.join(self.images_dir, f) 
                for f in os.listdir(self.images_dir) if f.endswith(".npy")
            ])
            self.mean = None
            self.std = None
            
           
            self.preprocessor = RAWDataPreProcessor(
                mean=self.mean, 
                std=self.std, 
                black_level=black_level, 
                white_level=white_level
            )
            
            logger.info(f"Created RGGBSegmentationDataset with {len(self.image_paths)} images")
            logger.info(f"Images dir: {self.images_dir}")
            logger.info(f"Masks dir: {self.masks_dir}")
            if len(self.image_paths) > 0:
                logger.info(f"First image path: {self.image_paths[0]}")

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            img_path = self.image_paths[idx]
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            mask_path = os.path.join(self.masks_dir, base_name + ".png")
            
            # Load RGGB .npy
            try:
                try:
                    raw_data = np.load(img_path, allow_pickle=True)
                except Exception:
                    print("Img path", img_path)
                    if self.split == "train":
                        raw_data = np.load(f"/home/paperspace/Documents/nika_space/main_dataset/val/ade/images/{base_name}.npy", allow_pickle=True)
                    else:
                        raw_data = np.load(f"/home/paperspace/Documents/nika_space/main_dataset/train/ade/images/{base_name}.npy", allow_pickle=True)
                    

                # Handle different types of .npy files
                if isinstance(raw_data, dict):
                    raw_image = raw_data.get('image') or raw_data.get('raw') or next(iter(raw_data.values()))
                else:
                    raw_image = raw_data
                
                # Ensure RGGB data is in the right format
                raw_image = raw_image.astype(np.float32)
                
                # Make sure we have the right dimensions [C, H, W]
                if raw_image.ndim == 2:  # If it's a 2D array, add channel dimension
                    raw_image = raw_image[np.newaxis, ...]
                elif raw_image.ndim == 3 and raw_image.shape[0] not in [1, 3, 4]:
                    # If channels are not in first dimension, rearrange to [C, H, W]
                    raw_image = raw_image.transpose(2, 0, 1)
                
                # Convert to tensor
                raw_tensor = torch.from_numpy(raw_image)
                
                # Preprocess
                processed = self.preprocessor({'image': raw_tensor})['image']
                
                # Apply additional transforms if provided
                if self.transform:
                    processed = self.transform(processed)
                
                # Load mask
                try:
                    mask = Image.open(mask_path).convert("L")
                except Exception:
                    if self.split == "train":
                        mask = Image.open(mask_path.replace("training", "validation")).convert("L")
                    else:
                        mask = Image.open(mask_path.replace("validation", "training")).convert("L")

                if self.target_transform:
                    mask = self.target_transform(mask)
                else:
                    mask = torch.from_numpy(np.array(mask)).long()
                
                return processed, mask
            
            except Exception as e:
                logger.error(f"Error loading image or mask: {e}")
                logger.error(f"Image path: {img_path}")
                logger.error(f"Mask path: {mask_path}")
                
                # Return zeros as a fallback
                dummy_image = torch.zeros((4, 224, 224), dtype=torch.float32)  # 4 channels for RGGB
                dummy_mask = torch.zeros((224, 224), dtype=torch.long)
                return dummy_image, dummy_mask

        def get_targets(self):
            class_set = set()
            for img_path in self.image_paths:
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                mask_path = os.path.join(self.masks_dir, base_name + ".png")
                try:
                    mask = np.array(Image.open(mask_path).convert("L"))
                    class_set.update(np.unique(mask))
                except Exception as e:
                    logger.warning(f"Could not load mask for {mask_path}: {e}")
            return sorted(list(class_set))
    
        


    mean = params_dict.get("mean")
    std = params_dict.get("std")
    black_level = float(params_dict.get("black_level", 0))
    white_level = float(params_dict.get("white_level", 1.0))
    
    return RGGBSegmentationDataset(
        root=params_dict.get("root"),
        split=split,
        transform=transform,
        target_transform=transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x * 255).long().squeeze(0))  # L-mode → single channel
        ]),
        mean=None,
        std=None,
        black_level=black_level,
        white_level=white_level
    )


def collate_fn(batch):
    """Custom collate function that properly handles RGGB data and segmentation masks"""
    images, masks = zip(*batch)
    
    images = torch.stack(images)

    masks = torch.stack(masks)
    
    return images, masks


def compute_metrics(preds, targets, num_classes):
    """Compute segmentation metrics"""
    preds = preds.argmax(dim=1)

    metrics = {
        "pixel_acc": 0.0,
        "mean_iou": 0.0,
        "class_iou": [0.0] * num_classes
    }

    valid_pixels = (targets != 255)
    correct_pixels = ((preds == targets) & valid_pixels)
    metrics["pixel_acc"] = correct_pixels.sum().float() / valid_pixels.sum().float()
 
    ious = []
    for cls in range(num_classes):
        pred_cls = (preds == cls)
        target_cls = (targets == cls)
        
        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()
        
        if union > 0:
            iou = intersection / union
            ious.append(iou)
            metrics["class_iou"][cls] = iou
        else:
            ious.append(torch.tensor(0.0, device=preds.device))
    

    metrics["mean_iou"] = sum(ious) / len(ious)
    
    return metrics


@torch.no_grad()
def evaluate_model(model, data_loader, device, num_classes):
    """Evaluate segmentation model"""
    model.eval()
    metrics_sum = {
        "pixel_acc": 0.0,
        "mean_iou": 0.0,
        "class_iou": [0.0] * num_classes
    }
    num_samples = 0
    
    for images, targets in data_loader:

        images = images.to(device)
        targets = targets.to(device)

        outputs = model(images)

        batch_metrics = compute_metrics(outputs, targets, num_classes)

        metrics_sum["pixel_acc"] += batch_metrics["pixel_acc"] * images.size(0)
        metrics_sum["mean_iou"] += batch_metrics["mean_iou"] * images.size(0)
        for cls in range(num_classes):
            metrics_sum["class_iou"][cls] += batch_metrics["class_iou"][cls] * images.size(0)
        
        num_samples += images.size(0)
   
        visualize_segmentation_predictions(
            model=model,
            data_loader=data_loader,
            device=torch.cuda.current_device(),
            output_dir=args.output_dir,
            num_samples=10
        )

    metrics = {
        "pixel_acc": metrics_sum["pixel_acc"] / num_samples,
        "mean_iou": metrics_sum["mean_iou"] / num_samples,
        "class_iou": [iou / num_samples for iou in metrics_sum["class_iou"]]
    }
    
    return metrics


def train_segmentation_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    loss_fn,
    num_classes,
    device,
    output_dir,
    max_iter,
    checkpoint_period,
    eval_period,
    start_iter=0,
):
    """Train segmentation model"""
    checkpointer = Checkpointer(model, output_dir, optimizer=optimizer, scheduler=scheduler)
    periodic_checkpointer = PeriodicCheckpointer(checkpointer, checkpoint_period, max_iter=max_iter)

    metric_logger = MetricLogger(delimiter="  ")
    header = "Training"

    model.train()
  
    iteration = start_iter
    logger.info(f"Starting training from iteration {start_iter} {max_iter}")
    
    for data, targets in metric_logger.log_every(
        train_loader,
        10,
        header,
        max_iter,
        start_iter,
    ):
        data = data.to(device)
        targets = targets.to(device)

        outputs = model(data)

        loss = loss_fn(outputs, targets)
  
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if iteration % 10 == 0:
            torch.cuda.synchronize()
            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        

        periodic_checkpointer.step(iteration)
        
        if eval_period > 0 and (iteration + 1) % eval_period == 0:
            metrics = evaluate_model(model, val_loader, device, num_classes)
            logger.info(f"ITER {iteration} - pixel_acc: {metrics['pixel_acc']:.4f}, mean_iou: {metrics['mean_iou']:.4f}")

            if distributed.is_main_process():
                metrics_file_path = os.path.join(output_dir, "segmentation_metrics.json")
                with open(metrics_file_path, "a") as f:
                    f.write(json.dumps({
                        "iteration": iteration,
                        "pixel_acc": metrics["pixel_acc"].item(),
                        "mean_iou": metrics["mean_iou"].item(),
                        "class_iou": [iou.item() if isinstance(iou, torch.Tensor) else iou for iou in metrics["class_iou"]]
                    }) + "\n")


            model.train()
        
        iteration += 1
        
        if iteration >= max_iter:
            break
 
    metrics = evaluate_model(model, val_loader, device, num_classes)
    logger.info(f"FINAL - pixel_acc: {metrics['pixel_acc']:.4f}, mean_iou: {metrics['mean_iou']:.4f}")
    
  
    if distributed.is_main_process():
        metrics_file_path = os.path.join(output_dir, "segmentation_metrics.json")
        with open(metrics_file_path, "a") as f:
            f.write(json.dumps({
                "iteration": iteration,
                "pixel_acc": metrics["pixel_acc"].item(),
                "mean_iou": metrics["mean_iou"].item(),
                "class_iou": [iou.item() if isinstance(iou, torch.Tensor) else iou for iou in metrics["class_iou"]]
            }) + "\n")
    
    return metrics


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class AttentionBlock(nn.Module):
    """Self-attention block for the decoder"""
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.key = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, C, H, W = x.size()

        proj_query = self.query(x).view(batch_size, -1, H * W).permute(0, 2, 1)
        proj_key = self.key(x).view(batch_size, -1, H * W)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=2)
        
        proj_value = self.value(x).view(batch_size, -1, H * W)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, H, W)
        
        out = self.gamma * out + x
        return out


class FeatureRefinementModule(nn.Module):
    """Module to refine features from DINOv2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)
        self.attention = AttentionBlock(out_channels)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        identity = x
        x = self.conv2(x)
        x = x + identity
        x = self.relu(x)
        x = self.attention(x)
        return x


class UpsampleBlock(nn.Module):
    """Enhanced upsampling block with residual connections"""
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, skip_features=None):
        x = self.upsample(x)
        
        if skip_features is not None:
            if x.shape[2:] != skip_features.shape[2:]:
                skip_features = F.interpolate(skip_features, size=x.shape[2:], mode='bilinear', align_corners=False)
            x = x + skip_features
            
        x = self.conv_block(x)
        return x


class ASPPModule(nn.Module):
    """Atrous Spatial Pyramid Pooling"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        rates = [1, 6, 12, 18]
        
        self.aspp_blocks = nn.ModuleList()
        for rate in rates:
            self.aspp_blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.output_conv = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):

        aspp_outputs = [block(x) for block in self.aspp_blocks]
  
        global_features = self.global_avg_pool(x)
        global_features = F.interpolate(global_features, size=x.shape[2:], mode='bilinear', align_corners=False)

        outputs = aspp_outputs + [global_features]
        outputs = torch.cat(outputs, dim=1)
        
        outputs = self.output_conv(outputs)
        return outputs


class EnhancedDINOv2SegmentationDecoder(nn.Module):
    """Enhanced decoder module for segmentation using DINOv2 features"""
    def __init__(self, dinov2_dim=768, hidden_dim=256, num_classes=21):
        super().__init__()
        

        self.feature_refine = FeatureRefinementModule(dinov2_dim, hidden_dim)
        
        self.aspp = ASPPModule(hidden_dim, hidden_dim)
        

        self.upsample1 = UpsampleBlock(hidden_dim, hidden_dim // 2)
        self.upsample2 = UpsampleBlock(hidden_dim // 2, hidden_dim // 4)
        self.upsample3 = UpsampleBlock(hidden_dim // 4, hidden_dim // 8)
        self.upsample4 = UpsampleBlock(hidden_dim // 8, hidden_dim // 16)
        
        self.classifier = nn.Sequential(
            nn.Conv2d(hidden_dim // 16, hidden_dim // 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim // 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 16, num_classes, kernel_size=1)
        )
        
        self.attention = AttentionBlock(hidden_dim // 16)

    def forward(self, features):

        x = self.feature_refine(features)
        

        x = self.aspp(x)

        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.upsample3(x)
        x = self.upsample4(x)
        
        x = self.attention(x)
        
        if x.size()[2:] != (224, 224):
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        x = self.classifier(x)
        
        return x


class EnhancedDINOv2SegmentationModel(nn.Module):
    """Enhanced segmentation model with DINOv2 encoder and advanced decoder"""
    def __init__(self, dinov2_model, num_classes=21, patch_size=14, img_size=224):
        super().__init__()
        # self.rggb_processor = RGGB2RGBModule(in_channels=4, out_channels=3)
        self.encoder = dinov2_model
        
 
        self.dinov2_dim = dinov2_model.embed_dim
        self.patch_size = patch_size
        self.img_size = img_size
        self.encoder.eval()
        

        self.decoder = EnhancedDINOv2SegmentationDecoder(
            dinov2_dim=self.dinov2_dim,
            hidden_dim=384,  # Increased from 256
            num_classes=num_classes
        )

        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize the weights of the custom modules"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def extract_intermediate_features(self, x):
        """Extract hierarchical features from DINOv2 encoder"""
        features = self.encoder.get_intermediate_layers(x, n=4)
            
        processed_features = []
        batch_size = x.shape[0]
            
        for idx, feature in enumerate(features):
            tokens = feature[0]
            num_tokens = tokens.shape[0]
            token_dim = tokens.shape[1]
            
            num_patches_per_image = num_tokens // batch_size
            h = w = int(np.sqrt(num_patches_per_image))
            
            patch_tokens = tokens.reshape(batch_size, h, w, token_dim)
            patch_tokens = patch_tokens.permute(0, 3, 1, 2)
            processed_features.append(patch_tokens)
            
        main_features = processed_features[-1]

        return main_features

    def forward(self, x):
        """Forward pass through encoder and decoder"""

        features = self.extract_intermediate_features(x)
        logits = self.decoder(features)
        
        return logits

def setup_segmentation_model(dinov2_model, args, config=None):
    """Set up segmentation model, optimizer, and scheduler"""
    print("Num classes: ", args.num_classes)
    model = EnhancedDINOv2SegmentationModel(
        dinov2_model=dinov2_model,
        num_classes=args.num_classes,
        patch_size=args.patch_size,
        img_size=args.img_size
    )
    

    if args.use_fsdp and config is not None:
        model.cfg = config
    

    model = model.cuda()

    if args.use_fsdp:
        model.prepare_for_distributed_training()
    

    if args.freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False
    

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    

    max_iter = args.epochs * (args.train_dataset_size // args.batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iter, eta_min=1e-6)

    loss_fn = SegmentationLoss()
    
    return model, optimizer, scheduler, loss_fn, max_iter


def run_segmentation(args):
    """Main function to run segmentation"""

    dinov2_model, config = setup_and_build_model(args)
    
    mean = args.rggb_mean
    std = args.rggb_std
    
    args.train_dataset_str += f":mean={mean}:std={std}:black_level={args.black_level}:white_level={args.white_level}"
    args.val_dataset_str += f":mean={mean}:std={std}:black_level={args.black_level}:white_level={args.white_level}"
    

    train_transform = make_segmentation_transform(split="train", img_size=args.img_size)
    val_transform = make_segmentation_transform(split="val", img_size=args.img_size)
    
    try:
        logger.info(f"Creating training dataset from: {args.train_dataset_str}")
        train_dataset = make_segmentation_dataset(args.train_dataset_str, train_transform, split="train")
        
        logger.info(f"Creating validation dataset from: {args.val_dataset_str}")
        val_dataset = make_segmentation_dataset(args.val_dataset_str, val_transform, split="val")
    except Exception as e:
        logger.error(f"Failed to create datasets: {e}")
        raise
    

    args.train_dataset_size = len(train_dataset)
    
    args.num_classes = len(train_dataset.get_targets())
    print("Train dataset size: ", args.train_dataset_size, args.num_classes)

    train_loader = make_data_loader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=True,
        persistent_workers=True if args.num_workers > 0 else False,
        collate_fn=collate_fn
    )
    print("Batch:", args.batch_size)
    val_loader = make_data_loader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler_type=SamplerType.DISTRIBUTED if distributed.is_enabled() else SamplerType.DEFAULT,
        shuffle=False,
        drop_last=True,
        persistent_workers=True if args.num_workers > 0 else False,
        collate_fn=collate_fn
    )
    

    model, optimizer, scheduler, loss_fn, max_iter = setup_segmentation_model(dinov2_model, args, config)

    if distributed.is_enabled() and not args.use_fsdp:
        model = DistributedDataParallel(model)
    

        checkpointer = Checkpointer(model, args.output_dir, optimizer=optimizer, scheduler=scheduler)
    
    checkpoint_path = os.path.join(args.output_dir, "last_checkpoint")
    start_iter = 0

    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            checkpoint_name = f.read().strip()
            logger.info(f"Resuming from checkpoint: {checkpoint_name}")
            checkpoint_data = checkpointer.load(os.path.join(args.output_dir,checkpoint_name))
            start_iter = checkpoint_data.get("iteration", 0)
    else:
        logger.info("No checkpoint found. Starting from iteration 0.")


    metrics = train_segmentation_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        num_classes=args.num_classes,
        device=torch.cuda.current_device(),
        output_dir=args.output_dir,
        max_iter=max_iter,
        checkpoint_period=args.save_checkpoint_frequency * (args.train_dataset_size // args.batch_size),
        eval_period=args.eval_period_iterations,
        start_iter=start_iter
    )
    
    logger.info(f"Training completed. Final mean IoU: {metrics['mean_iou']:.4f}")
    return metrics


def main(args):
    """Main function"""

    try:
        if distributed.is_available() and not distributed.is_enabled():
            distributed.init()
    except Exception as e:
        logger.warning(f"Failed to initialize distributed training: {e}")
        logger.warning("Falling back to single GPU training")
    
    torch.manual_seed(0)
    np.random.seed(0)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(args.output_dir, "training.log"))
        ]
    )
    
    logger.info(f"Arguments: {args}")

    metrics = run_segmentation(args)
    
    return 0


if __name__ == "__main__":
    description = "DINOv2 Segmentation with RGGB Support"
    args_parser = get_args_parser(description=description)
    args = args_parser.parse_args()
    sys.exit(main(args))