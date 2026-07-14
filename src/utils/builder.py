import torch
from torch.utils.data import DataLoader
from diffusers.optimization import get_scheduler
from accelerate.logging import get_logger

from ..models import load_model
from ..datasets import load_dataset
from .loss import LossFunction

logger = get_logger(__name__, log_level="INFO")


def build_dataloaders(args, accelerator):
    train_ds = load_dataset(args.dataset_name, **args.dataset_args[args.dataset_name])
    val_ds = load_dataset(args.val_dataset_name, **args.dataset_args[args.val_dataset_name])

    train_loader = DataLoader(train_ds, **args.dataloader)
    val_loader = DataLoader(val_ds, **args.val_dataloader)

    train_loader, val_loader = accelerator.prepare(train_loader, val_loader)
    steps_per_epoch = len(train_loader) // args.gradient_accumulation_steps
    if steps_per_epoch <= 0:
        raise ValueError(
            "steps_per_epoch is 0. Increase dataset size or reduce dataloader.batch_size, "
            "num_processes, or gradient_accumulation_steps."
        )

    logger.info(f"Training dataset {args.dataset_name} contains {len(train_ds):,} triplets, one epoch equals {steps_per_epoch} steps")
    logger.info(f"Validating dataset {args.val_dataset_name} contains {len(val_ds):,} triplets")
    return train_loader, val_loader, steps_per_epoch


def build_model_and_optim(args, accelerator, steps_per_epoch):
    model = load_model(args.model_name, **args.model_args)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Full model params: {total_params:,}")
    logger.info(f"Trainable params: {trainable_params:,}")
    logger.info(f"Trainable ratio: {100 * trainable_params / total_params:.2f}%")

    if args.pretrained_path:
        ckpt = torch.load(args.pretrained_path, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=False)
        logger.info(f"Pre-trained model loaded from {args.pretrained_path}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.base_lr, **args.optimizer)
    scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=steps_per_epoch * args.epochs,
    )
    loss_fn = LossFunction(**args.loss_function_args).to(accelerator.device)

    return accelerator.prepare(model, optimizer), scheduler, loss_fn
