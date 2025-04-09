# # Copyright (c) Meta Platforms, Inc. and affiliates.
# #
# # This source code is licensed under the Apache License, Version 2.0
# # found in the LICENSE file in the root directory of this source tree.

# import logging

# from torchvision import transforms

# from .transforms import (
#     GaussianBlur,
#     make_normalize_transform,
# )


# logger = logging.getLogger("dinov2")


# class DataAugmentationDINO(object):
#     def __init__(
#         self,
#         global_crops_scale,
#         local_crops_scale,
#         local_crops_number,
#         global_crops_size=224,
#         local_crops_size=96,
#     ):
#         self.global_crops_scale = global_crops_scale
#         self.local_crops_scale = local_crops_scale
#         self.local_crops_number = local_crops_number
#         self.global_crops_size = global_crops_size
#         self.local_crops_size = local_crops_size

#         logger.info("###################################")
#         logger.info("Using data augmentation parameters:")
#         logger.info(f"global_crops_scale: {global_crops_scale}")
#         logger.info(f"local_crops_scale: {local_crops_scale}")
#         logger.info(f"local_crops_number: {local_crops_number}")
#         logger.info(f"global_crops_size: {global_crops_size}")
#         logger.info(f"local_crops_size: {local_crops_size}")
#         logger.info("###################################")

#         # random resized crop and flip
#         self.geometric_augmentation_global = transforms.Compose(
#             [
#                 transforms.RandomResizedCrop(
#                     global_crops_size, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
#                 ),
#                 transforms.RandomHorizontalFlip(p=0.5),
#             ]
#         )

#         self.geometric_augmentation_local = transforms.Compose(
#             [
#                 transforms.RandomResizedCrop(
#                     local_crops_size, scale=local_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
#                 ),
#                 transforms.RandomHorizontalFlip(p=0.5),
#             ]
#         )

#         # color distorsions / blurring
#         # color_jittering = transforms.Compose(
#         #     [
#         #         transforms.RandomApply(
#         #             [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
#         #             p=0.8,
#         #         ),
#         #         transforms.RandomGrayscale(p=0.2),
#         #     ]
#         # )

#         global_transfo1_extra = GaussianBlur(p=1.0)

#         global_transfo2_extra = transforms.Compose(
#             [
#                 GaussianBlur(p=0.1),
#                 # transforms.RandomSolarize(threshold=128, p=0.2),
#             ]
#         )

#         local_transfo_extra = GaussianBlur(p=0.5)

#         # normalization
#         self.normalize = transforms.Compose(
#             [
#                 transforms.ToTensor(),
#                 # make_normalize_transform(),
#             ]
#         )

#         self.global_transfo1 = transforms.Compose([global_transfo1_extra, self.normalize])
#         self.global_transfo2 = transforms.Compose([global_transfo2_extra, self.normalize])
#         self.local_transfo = transforms.Compose([local_transfo_extra, self.normalize])

#     def __call__(self, image):
#         output = {}

#         # global crops:
#         im1_base = self.geometric_augmentation_global(image)
#         global_crop_1 = self.global_transfo1(im1_base)

#         im2_base = self.geometric_augmentation_global(image)
#         global_crop_2 = self.global_transfo2(im2_base)

#         output["global_crops"] = [global_crop_1, global_crop_2]

#         # global crops for teacher:
#         output["global_crops_teacher"] = [global_crop_1, global_crop_2]

#         # local crops:
#         local_crops = [
#             self.local_transfo(self.geometric_augmentation_local(image)) for _ in range(self.local_crops_number)
#         ]
#         output["local_crops"] = local_crops
#         output["offsets"] = ()

#         return output

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
