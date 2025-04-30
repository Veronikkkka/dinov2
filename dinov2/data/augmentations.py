import logging
import torch
from torchvision import transforms
import torchvision.transforms.functional as TF

from .transforms import (
    GaussianBlur,
    make_normalize_transform,
)

logger = logging.getLogger("dinov2")

class DataAugmentationDINO(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size

        logger.info("###################################")
        logger.info("Using tensor-based data augmentation:")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info("###################################")

        self.global_crop = transforms.RandomResizedCrop(
            global_crops_size, scale=global_crops_scale, interpolation=TF.InterpolationMode.BICUBIC
        )
        self.local_crop = transforms.RandomResizedCrop(
            local_crops_size, scale=local_crops_scale, interpolation=TF.InterpolationMode.BICUBIC
        )

        self.hflip = transforms.RandomHorizontalFlip(p=0.5)

        self.global_transfo1 = transforms.Compose([
            GaussianBlur(p=1.0),
            # make_normalize_transform(),  # optional
        ])

        self.global_transfo2 = transforms.Compose([
            GaussianBlur(p=0.1),
        ])

        self.local_transfo = transforms.Compose([
            GaussianBlur(p=0.5),
        ])

    def __call__(self, image_tensor):
        # image_tensor: [C, H, W], assumed to be float32 or normalized already

        output = {}

        # global crops
        im1 = self.hflip(self.global_crop(image_tensor))
        crop1 = self.global_transfo1(im1)

        im2 = self.hflip(self.global_crop(image_tensor))
        crop2 = self.global_transfo2(im2)

        output["global_crops"] = [crop1, crop2]
        output["global_crops_teacher"] = [crop1, crop2]

        # local crops
        local_crops = []
        for _ in range(self.local_crops_number):
            local = self.hflip(self.local_crop(image_tensor))
            local = self.local_transfo(local)
            local_crops.append(local)

        output["local_crops"] = local_crops
        output["offsets"] = ()

        return output
