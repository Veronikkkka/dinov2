# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
from typing import Dict, Optional

import torch
from torch import nn
from torchmetrics import MetricCollection

from dinov2.data import DatasetWithEnumeratedTargets, SamplerType, make_data_loader
import dinov2.distributed as distributed
from dinov2.logging import MetricLogger


logger = logging.getLogger("dinov2")


class ModelWithNormalize(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, samples):
        return nn.functional.normalize(self.model(samples), dim=1, p=2)


class ModelWithIntermediateLayers(nn.Module):
    def __init__(self, feature_model, n_last_blocks, autocast_ctx):
        super().__init__()
        self.feature_model = feature_model
        self.feature_model.eval()
        self.n_last_blocks = n_last_blocks
        self.autocast_ctx = autocast_ctx

    def forward(self, images):
        with torch.inference_mode():
            with self.autocast_ctx():
                features = self.feature_model.get_intermediate_layers(
                    images, self.n_last_blocks, return_class_token=True
                )
        return features


from torchvision.utils import save_image
import torchvision.transforms.functional as TF
import os
from PIL import Image, ImageDraw, ImageFont

def save_debug_images(inputs, preds, save_dir, class_mapping=None, count=5, prefix="batch"):
    os.makedirs(save_dir, exist_ok=True)
    inputs = inputs[:count].cpu()
    preds = preds[:count].cpu()

    # Unnormalization values for ImageNet
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    for i in range(len(inputs)):
        img = inputs[i]

        if img.shape[0] == 1 or img.shape[0] == 4:
            img = img[0:3]

        # Unnormalize
        # img = img * std + mean
        # img = torch.clamp(img, 0, 1)

        # Convert to PIL and draw
        img = TF.to_pil_image(img)
        draw = ImageDraw.Draw(img)
        label_text = f"{class_mapping[preds[i].item()]}" if class_mapping is not None else str(preds[i].item())
        draw.text((5, 5), label_text, fill="red")

        img.save(os.path.join(save_dir, f"{prefix}_{i}.png"))

@torch.inference_mode()
def evaluate(
    model: nn.Module,
    data_loader,
    postprocessors: Dict[str, nn.Module],
    metrics: Dict[str, MetricCollection],
    device: torch.device,
    criterion: Optional[nn.Module] = None,
):
    model.eval()
    if criterion is not None:
        criterion.eval()

    for metric in metrics.values():
        metric = metric.to(device)

    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"
    import random
    batch = random.randint(0, 10)
    print(batch)
    i_batch = 0
    for samples, targets, *_ in metric_logger.log_every(data_loader, 10, header):
        outputs = model(samples.to(device))
        targets = targets.to(device)

        if criterion is not None:
            loss = criterion(outputs, targets)
            metric_logger.update(loss=loss.item())

        # print(i_batch, batch)
        if i_batch == batch:  # Change number of batches to save if needed
            # for k, postprocessor in postprocessors.items():
            # print(postprocessors.keys())
            preds_logits = postprocessors["classifier_4_blocks_avgpool_True_lr_0_00003"](outputs, targets)["preds"]
            preds = preds_logits.argmax(dim=1)  # Ensure 'preds' is in your postprocessor output
            save_debug_images(
                samples, preds,
                save_dir="debug_outputs_linear",  # Customize this path
                class_mapping=None,             # Pass actual class_mapping if available
                prefix=f"batch{i_batch}"
            )


        for k, metric in metrics.items():
            metric_inputs = postprocessors[k](outputs, targets)
            metric.update(**metric_inputs)
        i_batch +=1

    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats: {metric_logger}")

    stats = {k: metric.compute() for k, metric in metrics.items()}
    metric_logger_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return metric_logger_stats, stats


def all_gather_and_flatten(tensor_rank):
    tensor_all_ranks = torch.empty(
        distributed.get_global_size(),
        *tensor_rank.shape,
        dtype=tensor_rank.dtype,
        device=tensor_rank.device,
    )
    tensor_list = list(tensor_all_ranks.unbind(0))
    torch.distributed.all_gather(tensor_list, tensor_rank.contiguous())
    return tensor_all_ranks.flatten(end_dim=1)


def extract_features(model, dataset, batch_size, num_workers, gather_on_cpu=False):
    dataset_with_enumerated_targets = DatasetWithEnumeratedTargets(dataset)
    sample_count = len(dataset_with_enumerated_targets)
    data_loader = make_data_loader(
        dataset=dataset_with_enumerated_targets,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler_type=SamplerType.DISTRIBUTED,
        drop_last=False,
        shuffle=False,
    )
    return extract_features_with_dataloader(model, data_loader, sample_count, gather_on_cpu)


@torch.inference_mode()
def extract_features_with_dataloader(model, data_loader, sample_count, gather_on_cpu=False):
    gather_device = torch.device("cpu") if gather_on_cpu else torch.device("cuda")
    metric_logger = MetricLogger(delimiter="  ")
    features, all_labels = None, None
    for samples, (index, labels_rank) in metric_logger.log_every(data_loader, 10):
        samples = samples.cuda(non_blocking=True)
        labels_rank = labels_rank.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)
        features_rank = model(samples).float()

        # init storage feature matrix
        if features is None:
            features = torch.zeros(sample_count, features_rank.shape[-1], device=gather_device)
            labels_shape = list(labels_rank.shape)
            labels_shape[0] = sample_count
            all_labels = torch.full(labels_shape, fill_value=-1, device=gather_device)
            logger.info(f"Storing features into tensor of shape {features.shape}")

        # share indexes, features and labels between processes
        index_all = all_gather_and_flatten(index).to(gather_device)
        features_all_ranks = all_gather_and_flatten(features_rank).to(gather_device)
        labels_all_ranks = all_gather_and_flatten(labels_rank).to(gather_device)

        # update storage feature matrix
        if len(index_all) > 0:
            features.index_copy_(0, index_all, features_all_ranks)
            all_labels.index_copy_(0, index_all, labels_all_ranks)

    logger.info(f"Features shape: {tuple(features.shape)}")
    logger.info(f"Labels shape: {tuple(all_labels.shape)}")

    assert torch.all(all_labels > -1)

    return features, all_labels
