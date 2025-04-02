# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import argparse
import logging
import math
import os
from functools import partial

from fvcore.common.checkpoint import PeriodicCheckpointer
import torch


import dinov2.distributed as distributed
from dinov2.fsdp import FSDPCheckpointer
from dinov2.logging import MetricLogger
from dinov2.utils.config import setup
from dinov2.utils.utils import CosineScheduler

from dinov2.train.ssl_meta_arch import SSLMetaArch
from dinov2.data.datasets.augmentation_rggb import create_adaptive_raw_pipeline


torch.backends.cuda.matmul.allow_tf32 = True  # PyTorch 1.12 sets this to False by default
logger = logging.getLogger("dinov2")


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("DINOv2 training", add_help=add_help)
    parser.add_argument("--config-file", default="dinov2/dinov2/configs/train/custom.yaml", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Whether to not attempt to resume from the checkpoint directory. ",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--eval", type=str, default="", help="Eval type to perform")
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        default="",
        type=str,
        help="Output directory to save logs and checkpoints",
    )

    return parser


def build_optimizer(cfg, params_groups):
    return torch.optim.AdamW(params_groups, betas=(cfg.optim.adamw_beta1, cfg.optim.adamw_beta2))


def build_schedulers(cfg):
    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    lr = dict(
        base_value=cfg.optim["lr"],
        final_value=cfg.optim["min_lr"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=cfg.optim["warmup_epochs"] * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value=0,
    )
    wd = dict(
        base_value=cfg.optim["weight_decay"],
        final_value=cfg.optim["weight_decay_end"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
    )
    momentum = dict(
        base_value=cfg.teacher["momentum_teacher"],
        final_value=cfg.teacher["final_momentum_teacher"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
    )
    teacher_temp = dict(
        base_value=cfg.teacher["teacher_temp"],
        final_value=cfg.teacher["teacher_temp"],
        total_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value=cfg.teacher["warmup_teacher_temp"],
    )

    lr_schedule = CosineScheduler(**lr)
    wd_schedule = CosineScheduler(**wd)
    momentum_schedule = CosineScheduler(**momentum)
    teacher_temp_schedule = CosineScheduler(**teacher_temp)
    last_layer_lr_schedule = CosineScheduler(**lr)

    last_layer_lr_schedule.schedule[
        : cfg.optim["freeze_last_layer_epochs"] * OFFICIAL_EPOCH_LENGTH
    ] = 0  # mimicking the original schedules

    logger.info("Schedulers ready.")

    return (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    )


def apply_optim_scheduler(optimizer, lr, wd, last_layer_lr):
    for param_group in optimizer.param_groups:
        is_last_layer = param_group["is_last_layer"]
        lr_multiplier = param_group["lr_multiplier"]
        wd_multiplier = param_group["wd_multiplier"]
        param_group["weight_decay"] = wd * wd_multiplier
        param_group["lr"] = (last_layer_lr if is_last_layer else lr) * lr_multiplier

from torch.distributed.fsdp import StateDictType

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import FullStateDictConfig, StateDictType

def do_test(cfg, model, iteration):
    # Ensure we are on the main process before saving
    if distributed.is_main_process():
        iterstring = str(iteration)
        eval_dir = os.path.join(cfg.train.output_dir, "eval", iterstring)
        os.makedirs(eval_dir, exist_ok=True)

        # Define full state dict config
        # full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

        # # Set FSDP state_dict type before calling state_dict()
        # with FSDP.state_dict_type(model.teacher, StateDictType.FULL_STATE_DICT, full_state_dict_config):
        #     new_state_dict = model.teacher.state_dict()

        from torch.distributed.fsdp import FullStateDictConfig, StateDictType, state_dict_type

        full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with state_dict_type(model, StateDictType.FULL_STATE_DICT, full_state_dict_config):
            checkpointer = FSDPCheckpointer(model, save_dir=cfg.train.output_dir)
            checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=not args.no_resume)

        # Save teacher checkpoint
        # teacher_ckp_path = os.path.join(eval_dir, "teacher_checkpoint.pth")
        # torch.save({"teacher": new_state_dict}, teacher_ckp_path)


def do_test(cfg, model, iteration):
    new_state_dict = model.teacher.state_dict()

    if distributed.is_main_process():
        iterstring = str(iteration)
        eval_dir = os.path.join(cfg.train.output_dir, "eval", iterstring)
        os.makedirs(eval_dir, exist_ok=True)
        # save teacher checkpoint
        teacher_ckp_path = os.path.join(eval_dir, "teacher_checkpoint.pth")
        torch.save({"teacher": new_state_dict}, teacher_ckp_path)


from torch.utils.tensorboard import SummaryWriter

# def do_train(cfg, model, resume=False):

#     from dinov2.data import SamplerType, make_data_loader, make_dataset
#     from dinov2.data import collate_data_and_cast, DataAugmentationDINO, MaskingGenerator
#     writer = SummaryWriter(log_dir="/home/paperspace/Documents/nika_space/dinov2/dinov2/tensorboard_logs/with_merged_blocks_small_lr")
#     model.train()
#     inputs_dtype = torch.half
#     fp16_scaler = model.fp16_scaler  # for mixed precision training

#     # setup optimizer
#     print("Cfg optim: ", cfg.optim)
#     optimizer = build_optimizer(cfg, model.get_params_groups())
#     # optimizer = build_optimizer(cfg, [p for p in model.parameters() if p.requires_grad])
#     # trainable_params = [p for p in model.parameters() if p.requires_grad]
#     # optimizer = torch.optim.AdamW(trainable_params, lr=lr)

#     (
#         lr_schedule,
#         wd_schedule,
#         momentum_schedule,
#         teacher_temp_schedule,
#         last_layer_lr_schedule,
#     ) = build_schedulers(cfg)

#     # checkpointer
#     checkpointer = FSDPCheckpointer(model, cfg.train.output_dir, optimizer=optimizer, save_to_disk=True)

#     start_iter = checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1

#     OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
#     max_iter = cfg.optim.epochs * OFFICIAL_EPOCH_LENGTH

#     periodic_checkpointer = PeriodicCheckpointer(
#         checkpointer,
#         period=3 * OFFICIAL_EPOCH_LENGTH,
#         max_iter=max_iter,
#         max_to_keep=3,
#     )

#     # setup data preprocessing

#     img_size = cfg.crops.global_crops_size
#     patch_size = cfg.student.patch_size
#     n_tokens = (img_size // patch_size) ** 2
#     mask_generator = MaskingGenerator(
#         input_size=(img_size // patch_size, img_size // patch_size),
#         max_num_patches=0.5 * img_size // patch_size * img_size // patch_size,
#     )

#     data_transform = DataAugmentationDINO(
#         cfg.crops.global_crops_scale,
#         cfg.crops.local_crops_scale,
#         cfg.crops.local_crops_number,
#         global_crops_size=cfg.crops.global_crops_size,
#         local_crops_size=cfg.crops.local_crops_size,
#     )

#     collate_fn = partial(
#         collate_data_and_cast,
#         mask_ratio_tuple=cfg.ibot.mask_ratio_min_max,
#         mask_probability=cfg.ibot.mask_sample_probability,
#         n_tokens=n_tokens,
#         mask_generator=mask_generator,
#         dtype=inputs_dtype,
#     )

#     # setup data loader

#     dataset = make_dataset(
#         dataset_str=cfg.train.dataset_path,
#         transform=data_transform,
#         target_transform=lambda _: (),
#     )
#     # sampler_type = SamplerType.INFINITE
#     sampler_type = SamplerType.SHARDED_INFINITE
#     data_loader = make_data_loader(
#         dataset=dataset,
#         batch_size=cfg.train.batch_size_per_gpu,
#         num_workers=cfg.train.num_workers,
#         shuffle=True,
#         seed=start_iter,  # TODO: Fix this -- cfg.train.seed
#         sampler_type=sampler_type,
#         sampler_advance=0,  # TODO(qas): fix this -- start_iter * cfg.train.batch_size_per_gpu,
#         drop_last=True,
#         collate_fn=collate_fn,
#     )

#     # training loop

#     iteration = start_iter

#     logger.info("Starting training from iteration {}".format(start_iter))
#     metrics_file = os.path.join(cfg.train.output_dir, "training_metrics.json")
#     metric_logger = MetricLogger(delimiter="  ", output_file=metrics_file)
#     header = "Training"

#     for data in metric_logger.log_every(
#         data_loader,
#         10,
#         header,
#         max_iter,
#         start_iter,
#     ):
#         current_batch_size = data["collated_global_crops"].shape[0] / 2
#         if iteration > max_iter:
#             return

#         # apply schedules

#         lr = lr_schedule[iteration]
#         wd = wd_schedule[iteration]
#         mom = momentum_schedule[iteration]
#         teacher_temp = teacher_temp_schedule[iteration]
#         last_layer_lr = last_layer_lr_schedule[iteration]
#         apply_optim_scheduler(optimizer, lr, wd, last_layer_lr)

#         # compute losses

#         optimizer.zero_grad(set_to_none=True)
#         loss_dict = model.forward_backward(data, teacher_temp=teacher_temp)

#         # clip gradients

#         if fp16_scaler is not None:
#             if cfg.optim.clip_grad:
#                 fp16_scaler.unscale_(optimizer)
#                 for v in model.student.values():
#                     v.clip_grad_norm_(cfg.optim.clip_grad)
#             fp16_scaler.step(optimizer)
#             fp16_scaler.update()
#         else:
#             if cfg.optim.clip_grad:
#                 for v in model.student.values():
#                     v.clip_grad_norm_(cfg.optim.clip_grad)
#             optimizer.step()

#         # perform teacher EMA update

#         model.update_teacher(mom)

#         # logging

#         if distributed.get_global_size() > 1:
#             for v in loss_dict.values():
#                 torch.distributed.all_reduce(v)
#         loss_dict_reduced = {k: v.item() / distributed.get_global_size() for k, v in loss_dict.items()}
        
#         if math.isnan(sum(loss_dict_reduced.values())):
#             logger.info("NaN detected")
#             raise AssertionError
#         losses_reduced = sum(loss for loss in loss_dict_reduced.values())

#         metric_logger.update(lr=lr)
#         metric_logger.update(wd=wd)
#         metric_logger.update(mom=mom)
#         metric_logger.update(last_layer_lr=last_layer_lr)
#         metric_logger.update(current_batch_size=current_batch_size)
#         metric_logger.update(total_loss=losses_reduced, **loss_dict_reduced)
#         writer.add_scalar('Loss/Total_Loss', losses_reduced, iteration)
#         writer.add_scalar('Learning_Rate', lr, iteration)
#         writer.add_scalar('Weight_Decay', wd, iteration)
#         writer.add_scalar('Momentum', mom, iteration)
#         # checkpointing and testing

#         if cfg.evaluation.eval_period_iterations > 0 and (iteration + 1) % cfg.evaluation.eval_period_iterations == 0:
#             do_test(cfg, model, f"training_{iteration}")
#             torch.cuda.synchronize()
#         periodic_checkpointer.step(iteration)

#         iteration = iteration + 1
#     metric_logger.synchronize_between_processes()
#     writer.close()
#     return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def get_state_dict(module, name):
    """Helper function to safely get state_dict() from a module."""
    if hasattr(module, "state_dict"):
        return module.state_dict()
    else:
        logger.warning(f"{name} does not have state_dict() and will be skipped in checkpointing.")
        return None

def do_train(cfg, model, resume=False):
    from dinov2.data import SamplerType, make_data_loader, make_dataset
    from dinov2.data import collate_data_and_cast, DataAugmentationDINO, MaskingGenerator
    writer = SummaryWriter(log_dir="/home/paperspace/Documents/nika_space/dinov2/dinov2/tensorboard_logs/main_dataset")
    model.train()
    inputs_dtype = torch.half
    fp16_scaler = model.fp16_scaler  # for mixed precision training


    # setup optimizer
    print("Cfg optim: ", cfg.optim)
    optimizer = build_optimizer(cfg, model.get_params_groups())

    (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    ) = build_schedulers(cfg)

    # checkpointer
    checkpointer = FSDPCheckpointer(model, cfg.train.output_dir, optimizer=optimizer, save_to_disk=True)

    start_iter = checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1

    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    max_iter = cfg.optim.epochs * OFFICIAL_EPOCH_LENGTH

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer,
        period=3 * OFFICIAL_EPOCH_LENGTH,
        max_iter=max_iter,
        max_to_keep=3,
    )

    # setup data preprocessing

    img_size = cfg.crops.global_crops_size
    patch_size = cfg.student.patch_size
    n_tokens = (img_size // patch_size) ** 2
    mask_generator = MaskingGenerator(
        input_size=(img_size // patch_size, img_size // patch_size),
        max_num_patches=0.5 * img_size // patch_size * img_size // patch_size,
    )

    data_transform = DataAugmentationDINO(
        cfg.crops.global_crops_scale,
        cfg.crops.local_crops_scale,
        cfg.crops.local_crops_number,
        global_crops_size=cfg.crops.global_crops_size,
        local_crops_size=cfg.crops.local_crops_size,
    )
    # data_transform = create_adaptive_raw_pipeline(
    #     global_crops_size=cfg.crops.global_crops_size,
    #     local_crops_size=cfg.crops.local_crops_size,
    #     global_crops_scale=cfg.crops.global_crops_scale,
    #     local_crops_scale=cfg.crops.local_crops_scale,
    #     local_crops_number=cfg.crops.local_crops_number,
    # )
    collate_fn = partial(
        collate_data_and_cast,
        mask_ratio_tuple=cfg.ibot.mask_ratio_min_max,
        mask_probability=cfg.ibot.mask_sample_probability,
        n_tokens=n_tokens,
        mask_generator=mask_generator,
        dtype=inputs_dtype,
    )

    # setup data loader

    dataset = make_dataset(
        dataset_str=cfg.train.dataset_path,
        transform=data_transform,
        target_transform=lambda _: (),
    )
    # sampler_type = SamplerType.INFINITE
    sampler_type = SamplerType.SHARDED_INFINITE
    data_loader = make_data_loader(
        dataset=dataset,
        batch_size=cfg.train.batch_size_per_gpu,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        seed=start_iter,  # TODO: Fix this -- cfg.train.seed
        sampler_type=sampler_type,
        sampler_advance=0,  # TODO(qas): fix this -- start_iter * cfg.train.batch_size_per_gpu,
        drop_last=True,
        collate_fn=collate_fn,
    )

    # training loop

    iteration = start_iter

    logger.info("Starting training from iteration {}".format(start_iter))
    metrics_file = os.path.join(cfg.train.output_dir, "training_metrics.json")
    metric_logger = MetricLogger(delimiter="  ", output_file=metrics_file)
    header = "Training"
    warmup_iterations = cfg.optim.warmup_epochs * OFFICIAL_EPOCH_LENGTH

    for data in metric_logger.log_every(
        data_loader,
        10,
        header,
        max_iter,
        start_iter,
    ):
        current_batch_size = data["collated_global_crops"].shape[0] / 2
        if iteration > max_iter:
            return

        # Unfreeze original DINO weights after warmup
        if iteration == warmup_iterations:
            for param in model.parameters():
                param.requires_grad = True

   
        # apply schedules

        lr = lr_schedule[iteration]
        wd = wd_schedule[iteration]
        mom = momentum_schedule[iteration]
        teacher_temp = teacher_temp_schedule[iteration]
        last_layer_lr = last_layer_lr_schedule[iteration]
        apply_optim_scheduler(optimizer, lr, wd, last_layer_lr)

        # compute losses

        optimizer.zero_grad(set_to_none=True)
        loss_dict = model.forward_backward(data, teacher_temp=teacher_temp)

        # clip gradients

        if fp16_scaler is not None:
            if cfg.optim.clip_grad:
                fp16_scaler.unscale_(optimizer)
                for v in model.student.values():
                    v.clip_grad_norm_(cfg.optim.clip_grad)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()
        else:
            if cfg.optim.clip_grad:
                for v in model.student.values():
                    v.clip_grad_norm_(cfg.optim.clip_grad)
            optimizer.step()

        # perform teacher EMA update
        model.update_teacher(mom)

        # logging

        if distributed.get_global_size() > 1:
            for v in loss_dict.values():
                torch.distributed.all_reduce(v)
        loss_dict_reduced = {k: v.item() / distributed.get_global_size() for k, v in loss_dict.items()}
        
        if math.isnan(sum(loss_dict_reduced.values())):
            logger.info("NaN detected")
            raise AssertionError
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        metric_logger.update(lr=lr)
        metric_logger.update(wd=wd)
        metric_logger.update(mom=mom)
        metric_logger.update(last_layer_lr=last_layer_lr)
        metric_logger.update(current_batch_size=current_batch_size)
        metric_logger.update(total_loss=losses_reduced, **loss_dict_reduced)
        writer.add_scalar('Loss/Total_Loss', losses_reduced, iteration)
        writer.add_scalar('Learning_Rate', lr, iteration)
        writer.add_scalar('Weight_Decay', wd, iteration)
        writer.add_scalar('Momentum', mom, iteration)
        # checkpointing and testing

        if cfg.evaluation.eval_period_iterations > 0 and (iteration + 1) % cfg.evaluation.eval_period_iterations == 0:
            do_test(cfg, model, f"training_{iteration}")
            torch.cuda.synchronize()
        periodic_checkpointer.step(iteration)

        iteration = iteration + 1
    metric_logger.synchronize_between_processes()
    writer.close()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

import re
from typing import List, Union

def parse_merge_block_indexes(config_value: str) -> List[int]:
    """
    Parses a string containing merge block indexes and returns a list of integers.
    Supports formats like "1,3,7,11" or "0..11".
    """
    if '..' in config_value:
        start, end = map(int, config_value.split('..'))
        return list(range(start, end + 1))
    if config_value == "":
        return []
    return list(map(int, re.split(r'\s*,\s*', config_value)))

def main(args):
    cfg = setup(args)
    print("cfg.student.merge_block_indexes ", cfg.student.merge_block_indexes)
    cfg.student.merge_block_indexes = parse_merge_block_indexes(cfg.student.merge_block_indexes)
    print("cfg.student.merge_block_indexes:", cfg.student.merge_block_indexes)

    model = SSLMetaArch(cfg).to(torch.device("cuda"))
    model.prepare_for_distributed_training()
    
    # model.freeze_original_dino_weights()
    
    if args.eval_only:
        iteration = (
            FSDPCheckpointer(model, save_dir=cfg.train.output_dir)
            .resume_or_load(cfg.MODEL.WEIGHTS, resume=not args.no_resume)
            .get("iteration", -1)
            + 1
        )
        return do_test(cfg, model, f"manual_{iteration}")
    
    do_train(cfg, model, resume=not args.no_resume)


if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    main(args)
