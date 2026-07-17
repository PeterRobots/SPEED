import math

import torch
from torch.utils.data import DataLoader
from diffusers.optimization import get_scheduler
from accelerate.logging import get_logger

from ..datasets import load_dataset
from ..runtime import build_model_from_config
from .loss import LossFunction

logger = get_logger(__name__, log_level="INFO")


def build_dataloaders(args, accelerator):
    train_ds = load_dataset(args.dataset_name, **args.dataset_args[args.dataset_name])
    val_ds = load_dataset(args.val_dataset_name, **args.dataset_args[args.val_dataset_name])

    train_loader = DataLoader(train_ds, **args.dataloader)
    val_loader = DataLoader(val_ds, **args.val_dataloader)

    train_loader, val_loader = accelerator.prepare(train_loader, val_loader)
    steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    if steps_per_epoch <= 0:
        raise ValueError(
            "steps_per_epoch is 0. Increase dataset size or reduce dataloader.batch_size, "
            "num_processes, or gradient_accumulation_steps."
        )

    logger.info(f"Training dataset {args.dataset_name} contains {len(train_ds):,} triplets, one epoch equals {steps_per_epoch} steps")
    logger.info(f"Validating dataset {args.val_dataset_name} contains {len(val_ds):,} triplets")
    return train_loader, val_loader, steps_per_epoch


def build_model_and_optim(args, accelerator, total_steps):
    model, model_info = build_model_from_config(args, strict_load=False)
    total_params = model_info["total_params"]
    trainable_params = model_info["trainable_params"]
    logger.info(f"Full model params: {total_params:,}")
    logger.info(f"Trainable params: {trainable_params:,}")
    logger.info(f"Trainable ratio: {100 * trainable_params / total_params:.2f}%")

    if model_info["checkpoint_path"]:
        if model_info["missing_keys"]:
            logger.warning(f"Missing checkpoint keys: {len(model_info['missing_keys'])}")
        if model_info["unexpected_keys"]:
            logger.warning(f"Unexpected checkpoint keys: {len(model_info['unexpected_keys'])}")
        logger.info(f"Pre-trained model loaded from {model_info['checkpoint_path']}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.base_lr, **args.optimizer)
    scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )
    loss_fn = LossFunction(**args.loss_function_args).to(accelerator.device)

    return accelerator.prepare(model, optimizer), scheduler, loss_fn
