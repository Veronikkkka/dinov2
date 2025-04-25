# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from typing import Sequence

import torch
from torchvision import transforms


class GaussianBlur(transforms.RandomApply):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(self, *, p: float = 0.5, radius_min: float = 0.1, radius_max: float = 2.0):
        # NOTE: torchvision is applying 1 - probability to return the original image
        keep_p = 1 - p
        transform = transforms.GaussianBlur(kernel_size=9, sigma=(radius_min, radius_max))
        super().__init__(transforms=[transform], p=keep_p)


class MaybeToTensor(transforms.ToTensor):
    """
    Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor, or keep as is if already a tensor.
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image, numpy.ndarray or torch.tensor): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, torch.Tensor):
            return pic
        return super().__call__(pic)


# Use timm's names
IMAGENET_DEFAULT_MEAN = (0.0446893610060215, 0.11062706261873245, 0.11063370108604431, 0.06802941858768463)  #(0.485, 0.456, 0.406) #
IMAGENET_DEFAULT_STD = (0.0667753592133522, 0.12049347162246704, 0.12050503492355347, 0.07335837930440903) #(0.229, 0.224, 0.225) #


def make_normalize_transform(
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> transforms.Normalize:
    return transforms.Normalize(mean=mean, std=std)


# This roughly matches torchvision's preset for classification training:
#   https://github.com/pytorch/vision/blob/main/references/classification/presets.py#L6-L44
def make_classification_train_transform(
    *,
    crop_size: int = 224,
    interpolation=transforms.InterpolationMode.BILINEAR,
    hflip_prob: float = 0.5,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
):
    transforms_list = [
        transforms.RandomResizedCrop(crop_size, interpolation=interpolation)
    ]
    
    if hflip_prob > 0.0:
        transforms_list.append(transforms.RandomHorizontalFlip(hflip_prob))
        
    # No need for MaybeToTensor() if input is already a tensor
    transforms_list.append(
        transforms.Normalize(mean=mean, std=std)
    )
    
    return transforms.Compose(transforms_list)


# This matches (roughly) torchvision's preset for classification evaluation:
#   https://github.com/pytorch/vision/blob/main/references/classification/presets.py#L47-L69
def make_classification_eval_transform(
    *,
    resize_size: int = 256,
    interpolation=transforms.InterpolationMode.BILINEAR,
    crop_size: int = 224,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> transforms.Compose:
    transforms_list = [
        transforms.Resize(resize_size, interpolation=interpolation),
        transforms.CenterCrop(crop_size),
        MaybeToTensor(),
        # make_normalize_transform(mean=mean, std=std),
    ]
    return transforms.Compose(transforms_list)

# def make_classification_eval_transform(
#     *,
#     resize_size: int = 256,
#     interpolation=transforms.InterpolationMode.BICUBIC,
#     crop_size: int = 224,
#     mean: Sequence[float] = [0.485, 0.456, 0.406],  # Default for ImageNet
#     std: Sequence[float] = [0.229, 0.224, 0.225],   # Default for ImageNet
# ) -> transforms.Compose:
#     transforms_list = [
#         transforms.Resize(resize_size, interpolation=interpolation),
#         transforms.CenterCrop(crop_size),
#         ToTensor4Channels(),  # Custom transformation for 4-channel input
#         transforms.Normalize(mean=mean, std=std),  # Apply normalization
#     ]
#     return transforms.Compose(transforms_list)


# class ToTensor4Channels(object):
#     def __call__(self, sample):
#         # Assumes input is [4, H, W] - a tensor with 4 channels
#         if isinstance(sample, torch.Tensor):
#             # You can permute the tensor to [H, W, 4] for standard image processing
#             sample = sample.permute(1, 2, 0)  # Convert from [4, H, W] to [H, W, 4]
            
#             # Normalize to [0, 1] range (you can also do min-max normalization if required)
#             sample = sample.float() / 255.0
            
#             # Now sample is a [H, W, 4] tensor; you may apply further processing for RGGB, etc.
#             return torch.tensor(sample, dtype=torch.float32)
        
#         return sample
    
# def make_classification_eval_transform(
#     *,
#     resize_size: int = 256,
#     crop_size: int = 224,
#     mean: Sequence[float] = [0.5, 0.5, 0.5, 0.5],  # Use appropriate values for your RGGB data
#     std: Sequence[float] = [0.5, 0.5, 0.5, 0.5],   # Use appropriate values for your RGGB data
# ) -> transforms.Compose:
#     transforms_list = [
#         # Assumes input is already a tensor in [4, H, W] format
#         transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BILINEAR),
#         transforms.CenterCrop(crop_size),
#         transforms.Normalize(mean=mean, std=std),
#     ]
#     return transforms.Compose(transforms_list)