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
    print("In visualization")
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    model.eval()
    
    colormap = torch.randint(0, 256, (256, 3), dtype=torch.uint8)
   
    colormap[0] = torch.tensor([0, 0, 0], dtype=torch.uint8)
    print("Num samples: ", num_samples)
    processed_samples = 0
    with torch.no_grad():
        for i, (images, targets) in enumerate(data_loader):

            if i >= num_samples:
                break
                
            images = images.to(device)
            targets = targets.to(device)
            

            outputs = model(images)

            preds = outputs.argmax(dim=1)

            for j in range(images.shape[0]):

                if i*data_loader.batch_size + j >= num_samples:
                    break
                    
                image = images[j].cpu()
                print("Image: ", image.shape, image.min(), image.max())
                target = targets[j].cpu()
                unique_target, counts_target = torch.unique(target, return_counts=True)
                
                pred = preds[j].cpu()
                unique_pred, counts_pred = torch.unique(pred, return_counts=True)
                print("GT: ", unique_target, counts_target)
                print("Unique: ", unique_pred, counts_pred)

                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                

                if image.shape[0] == 4:

                    rgb_image = torch.stack([
                        image[0],
                        (image[1] + image[2]) / 2,
                        image[3]
                    ], dim=0)

                    rgb_min = rgb_image.min()
                    rgb_max = rgb_image.max()
                    rgb_image = (rgb_image - rgb_min) / (rgb_max - rgb_min)
                    print("RGB image:", type(rgb_image), rgb_image.shape)
                    
                    axes[0].imshow(rgb_image.permute(1, 2, 0).numpy())
                else:

                    axes[0].imshow(image.permute(1, 2, 0).numpy())
                axes[0].set_title("Input Image")
                axes[0].axis('off')
                

                target_vis = colormap[target].permute(2, 0, 1)
                axes[1].imshow(target_vis.permute(1, 2, 0).numpy())
                axes[1].set_title("Ground Truth")
                axes[1].axis('off')

                pred_vis = colormap[pred].permute(2, 0, 1)
                axes[2].imshow(pred_vis.permute(1, 2, 0).numpy())
                axes[2].set_title("Prediction")
                axes[2].axis('off')
                

                sample_idx = i*data_loader.batch_size + j
                plt.tight_layout()
                plt.savefig(os.path.join(vis_dir, f"segmentation_sample_{sample_idx}.png"))
                plt.close(fig)

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
                processed_samples += 1
            if processed_samples >= num_samples:
                break
                
    
    print(f"Saved visualizations to {vis_dir}")

class UpsampleBlock(nn.Module):
    """Upsampling block for decoder"""
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x



class DINOv2SegmentationModel(nn.Module):
    def __init__(self, dinov2_model, num_classes=21, patch_size=14, img_size=224, input_format="rgb"):
        super().__init__()
        self.encoder = dinov2_model
        self.dinov2_dim = dinov2_model.embed_dim
        self.patch_size = patch_size
        self.img_size = img_size
        self.input_format = input_format
        

        for name, param in self.encoder.named_parameters():
            if "blocks.10" in name or "blocks.11" in name or "norm" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.decoder = DINOv2SegmentationDecoder(
            dinov2_dim=self.dinov2_dim,
            hidden_dim=256,
            num_classes=num_classes
        )

        if input_format == "rggb":
            self.rggb_processor = RGGB2RGBModule(in_channels=4, out_channels=3)
        else:
            self.rggb_processor = None

    def forward(self, x):

        if self.input_format == "rggb" and x.shape[1] == 4 and self.rggb_processor is not None:
            x = self.rggb_processor(x)

        features = self.encoder(x, is_training=True)
        patch_tokens = features["x_norm_patchtokens"]  # [B, N, C]
        B, N, C = patch_tokens.shape
        H = W = int(N ** 0.5)
        patch_tokens = patch_tokens.transpose(1, 2).reshape(B, C, H, W)
 
        logits = self.decoder(patch_tokens)
        
        return logits


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
        num_classes=151,
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
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    
    if split == "train":
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.2),
            ToTensorV2()
        ])

    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            ToTensorV2()
        ])



def get_segmentation_optimizer_scheduler(model, max_iter):
    encoder_params = list(model.encoder.parameters())
    decoder_params = list(model.decoder.parameters())
    
    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": 1e-6},
        {"params": decoder_params, "lr": 1e-3}
    ], weight_decay=0.01)

    warmup_iters = max_iter // 10
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[1e-6, 1e-3],
        total_steps=max_iter,
        pct_start=0.1,
        anneal_strategy='cos'
    )
    
    return optimizer, scheduler

import cv2
import numpy as np

def rggb_to_rgb(npy_path):
    import cv2
    array = np.load(npy_path)  

    if array.dtype != np.uint8:
        if array.max() > 1:
            array = (array / array.max() * 255).astype(np.uint8)
        else:
            array = (array * 255).astype(np.uint8)

    bayer_mosaic = pack_rggb_planes_to_bayer(array)
    rgb = cv2.cvtColor(bayer_mosaic, cv2.COLOR_BAYER_RG2RGB)
    
    return rgb


import numpy as np
from PIL import Image

def visualize_rggb_image(tensor, output_path="check_9.png"):
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

    def robust_normalize(arr):
        lower = np.percentile(arr, 2)
        upper = np.percentile(arr, 98)
        if upper == lower:
            return np.zeros_like(arr)
        arr_clipped = np.clip(arr, lower, upper)
        return (arr_clipped - lower) / (upper - lower)

    r_normalized = robust_normalize(r)
    g1_normalized = robust_normalize(g1)
    g2_normalized = robust_normalize(g2)
    b_normalized = robust_normalize(b)

    g = (g1_normalized + g2_normalized) / 2

    rgb = np.stack([r_normalized, g, b_normalized], axis=2)
    rgb_normalized = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    
    Image.fromarray(rgb_normalized).save(output_path)
    
    return rgb_normalized



import random

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
            
            try:
                try:
                    raw_data = np.load(img_path, allow_pickle=True)
                except Exception:

                    if self.split == "train":
                        raw_data = np.load(f"/home/paperspace/Documents/nika_space/main_dataset/val/ade/images/{base_name}.npy", allow_pickle=True)
                    else:
                        raw_data = np.load(f"/home/paperspace/Documents/nika_space/main_dataset/train/ade/images/{base_name}.npy", allow_pickle=True)
                    

                if isinstance(raw_data, dict):
                    raw_image = raw_data.get('image') or raw_data.get('raw') or next(iter(raw_data.values()))
                else:
                    raw_image = raw_data
                
                raw_image = raw_image.astype(np.float32)
                
                if raw_image.ndim == 2: 
                    raw_image = raw_image[np.newaxis, ...]
                elif raw_image.ndim == 3 and raw_image.shape[0] not in [1, 3, 4]:
                    raw_image = raw_image.transpose(2, 0, 1)
                
                raw_tensor = torch.from_numpy(raw_image)
                
                processed = self.preprocessor({'image': raw_tensor})['image']
                mask = Image.open(mask_path).convert("L")
                # [C, H, W]
                height, width = processed.shape[-2], processed.shape[-1]
                mask = mask.resize((width, height), Image.NEAREST)

                mask = np.array(mask)
                processed_np = processed.numpy()
                if processed_np.ndim == 3 and processed_np.shape[0] in [1, 3, 4]:
                    # from (C, H, W) to (H, W, C)
                    processed_np = np.transpose(processed_np, (1, 2, 0))

                mask_np = mask.numpy() if isinstance(mask, torch.Tensor) else mask

                if self.transform:
                    augmented = self.transform(image=processed_np, mask=mask_np)
                    processed = augmented['image'].float()
                    mask = augmented['mask'].long()

                visualization_image = visualize_rggb_image(processed.permute(1, 2, 0))
                Image.fromarray(visualization_image).save("check_9.png")

                return processed, mask
            
            except Exception as e:
                raise e

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
    processed = 0
    print("before cycle: ", len(data_loader))
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
        if processed > 5:
            break

        visualize_segmentation_predictions(
            model=model,
            data_loader=data_loader,
            device=torch.cuda.current_device(),
            output_dir=args.output_dir,
            num_samples=10
        )
        processed += 10

    metrics = {
        "pixel_acc": metrics_sum["pixel_acc"],
        "mean_iou": metrics_sum["mean_iou"],
        "class_iou": [iou for iou in metrics_sum["class_iou"]]
    }
    
    return metrics


class DynamicSegmentationLoss(nn.Module):
    def __init__(self, class_weights=None, ignore_index=255, max_iter=1000):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base_weights = class_weights # if class_weights else torch.ones(151)
        self.base_weights = self.base_weights.to(device)
        self.ignore_index = ignore_index
        self.current_step = 0
        self.max_steps = max_iter

    def step(self):
        self.current_step += 1

    def forward(self, preds, targets):

        decay = 1.0 - (self.current_step / self.max_steps)
        weights = 1.0 + (self.base_weights - 1.0) * decay
        ce_loss = nn.CrossEntropyLoss(weight=weights, ignore_index=self.ignore_index)
        return ce_loss(preds, targets)


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
    max_grad_norm=1.0
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
        outputs = outputs.to(device)
        targets = targets.to(device)

        loss_fn.step()
        loss = loss_fn(outputs, targets)
        
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        
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

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, ignore_index=255):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index
        
    def forward(self, inputs, targets):

        logp = F.log_softmax(inputs, dim=1)
        probs = torch.exp(logp)
        
        valid_mask = (targets != self.ignore_index).float()
        
        targets_one_hot = F.one_hot(
            targets.clamp(0), num_classes=inputs.shape[1]
        ).permute(0, 3, 1, 2).float()
        

        focal_weight = (1 - probs) ** self.gamma
        loss = -self.alpha * focal_weight * logp
        
        loss = loss * targets_one_hot * valid_mask.unsqueeze(1)
        
        return loss.sum() / (valid_mask.sum() + 1e-10)



def compute_class_weights(dataset, num_classes, min_weight=0.1, max_weight=5.0):
    class_pixel_counts = torch.ones(num_classes)

    for _, target in dataset:
        target = target.numpy()
        counts = np.bincount(target.flatten(), minlength=num_classes)
        class_pixel_counts += counts

    frequencies = class_pixel_counts / class_pixel_counts.sum()
    median_freq = np.median(frequencies[frequencies > 0])
    raw_weights = median_freq / (frequencies + 1e-6)


    raw_weights = np.clip(raw_weights, min_weight, max_weight)

    return torch.tensor(raw_weights, dtype=torch.float32)



class RGGB2RGBModule(nn.Module):
    """Module to convert RGGB features to RGB-like features for better compatibility with DINOv2"""
    def __init__(self, in_channels=4, out_channels=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=1)
        )
        
    def forward(self, x):
        return self.conv(x)



class DINOv2SegmentationDecoder(nn.Module):
    def __init__(self, dinov2_dim=768, hidden_dim=256, num_classes=150):
        super().__init__()
        
        self.project = nn.Conv2d(dinov2_dim, hidden_dim, kernel_size=1)
        
        self.upsample1 = UpsampleBlock(hidden_dim, hidden_dim // 2)
        self.upsample2 = UpsampleBlock(hidden_dim // 2, hidden_dim // 2)  # Keep more channels
        self.upsample3 = UpsampleBlock(hidden_dim // 2, hidden_dim // 4)
        
        self.refine = nn.Sequential(
            nn.Conv2d(hidden_dim // 4, hidden_dim // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim // 4),
            nn.ReLU(inplace=True)
        )
        
        self.classifier = nn.Conv2d(hidden_dim // 4, num_classes, kernel_size=1)
        nn.init.zeros_(self.classifier.bias)

    
    def forward(self, features):

        x = self.project(features)
        

        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.upsample3(x)
        
        x = self.refine(x)

        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        x = self.classifier(x)
        
        return x


def setup_segmentation_model(dinov2_model, args, config=None, class_weights=None):
    """Set up segmentation model, optimizer, and scheduler"""
    model = DINOv2SegmentationModel(
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
   
    # max_iter = args.epochs * (args.train_dataset_size // args.batch_size)
    max_iter = 50000
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iter, eta_min=1e-6)
    optimizer, scheduler = get_segmentation_optimizer_scheduler(model, max_iter)
    
    loss_fn = DynamicSegmentationLoss(class_weights=class_weights, max_iter=max_iter)
    
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
        print("Val dataset", len(val_dataset))
    except Exception as e:
        logger.error(f"Failed to create datasets: {e}")
        raise
    
    args.train_dataset_size = len(train_dataset)
    
    args.num_classes = int(torch.max(torch.tensor(train_dataset.get_targets())).item()) + 1
    print("Train dataset size: ", args.train_dataset_size, args.num_classes)

    train_loader = make_data_loader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
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
    
    class_weights = compute_class_weights(train_dataset, args.num_classes)
    model, optimizer, scheduler, loss_fn, max_iter = setup_segmentation_model(dinov2_model, args, config, class_weights)
    
    if distributed.is_enabled() and not args.use_fsdp:
        model = DistributedDataParallel(model, find_unused_parameters=True)
    

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
        checkpoint_period=5000,
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