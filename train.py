import os
import argparse
import warnings
from collections import defaultdict
from omegaconf import OmegaConf

import torch
import swanlab
from tqdm import tqdm
from torchvision.utils import save_image
from accelerate.utils import set_seed
from accelerate.logging import get_logger

from src.utils import CalMetrics, Trainer
from src.utils.loss import LossMeter
from src.utils.setup_utils import setup_accelerator, setup_experiment_dirs
from src.utils.builder import build_dataloaders, build_model_and_optim

warnings.filterwarnings("ignore")
logger = get_logger(__name__, log_level="INFO")

LOWER_METRICS = {"LPIPS", "FloLPIPS", "L1"}
HIGHER_METRICS = {"PSNR", "SSIM"}
SUPPORTED_METRICS = LOWER_METRICS | HIGHER_METRICS


def is_better(metric, value, best):
    return (
        (metric in LOWER_METRICS and value <= best) or
        (metric in HIGHER_METRICS and value >= best)
    )


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    args = parser.parse_args()
    return OmegaConf.load(args.config)


def run_validation(
    model,
    val_loader,
    trainer,
    metrics,
    accelerator,
    step,
    exp_dir,
):
    logger.info("Validating ...")
    model.eval()
    unwrapped_model = accelerator.unwrap_model(model)
    results = defaultdict(list)

    with torch.no_grad():
        for i, val_batch in enumerate(val_loader):
            pred, gt = trainer.validate_one_step(unwrapped_model, val_batch)
            pred = (pred / 2 + 0.5).clamp(0, 1)
            gt = (gt / 2 + 0.5).clamp(0, 1)

            metrics_step = {
                "PSNR": metrics.cal_psnr(pred, gt),
                "SSIM": metrics.cal_ssim(pred, gt),
                "LPIPS": metrics.cal_lpips(pred, gt),
                "FloLPIPS": metrics.cal_flolpips(pred, gt, val_batch[:, 0] / 255., val_batch[:, 1] / 255.),
                "L1": torch.mean(torch.abs(pred - gt), dim=(1, 2, 3)),
            }

            for k, v in metrics_step.items():
                gathered = accelerator.gather_for_metrics(v)
                results[k].extend(gathered.cpu().tolist())

            if accelerator.is_main_process and i < 2:
                save_dir = f"{exp_dir}/validation_results/steps{step:07d}"
                os.makedirs(save_dir, exist_ok=True)
                save_image(torch.stack((pred, gt), 1).cpu().flatten(0, 1), f"{save_dir}/val_{i}.jpg", nrow=8)

    model.train()

    final_results = {k: sum(v) / len(v) for k, v in results.items()}

    format_results = (
        f"PSNR: {final_results['PSNR']:.4f}, "
        f"SSIM: {final_results['SSIM']:.4f}, "
        f"LPIPS: {final_results['LPIPS']:.4f}, "
        f"FloLPIPS: {final_results['FloLPIPS']:.4f}, "
        f"L1: {final_results['L1']:.4f}"
    )
    if accelerator.is_main_process:
        with open(f"{exp_dir}/validation_results/val_results.txt", "a+", encoding="utf-8") as f:
            f.write(f"-*- Steps{step:06d} -*- {final_results}\n")

    logger.info(f"Steps{step:06d} validation results on DAVIS: {format_results}")
    return final_results


def log_train_visualization(
    trainer,
    frame_data_list,
    step,
    exp_dir,
    accelerator,
):
    if not accelerator.is_main_process:
        return

    visualization_list, nrow = trainer.visualize_frame_data(frame_data_list)
    swan_images = []

    for i, vis in enumerate(visualization_list):
        save_path = f"{exp_dir}/visualization_results/steps{step:07d}_gt_pred_{i}.jpg"
        save_image(vis, save_path, nrow=nrow)
        swan_images.append(swanlab.Image(save_path))
    accelerator.log({"visualization": swan_images}, step=step)


def train_loop(args, accelerator, exp_dir):
    if args.validate_metric not in SUPPORTED_METRICS:
        supported = ", ".join(sorted(SUPPORTED_METRICS))
        raise ValueError(f"Unsupported validate_metric: {args.validate_metric}. Supported metrics are: {supported}")

    train_loader, val_loader, steps_per_epoch = build_dataloaders(args, accelerator)
    (model, optimizer), scheduler, loss_fn = build_model_and_optim(args, accelerator, steps_per_epoch)

    trainer = Trainer(loss_fn, **args.trainer_args, device=accelerator.device)
    metrics = CalMetrics(accelerator.device)
    meter = LossMeter()

    best_metric = args.init_val_metric_value
    total_steps = steps_per_epoch * args.epochs
    logger.info(f"Training for {args.epochs} epochs ({total_steps} steps)...")
    step = 0

    if accelerator.is_main_process:
        tracker_config = OmegaConf.to_container(args, resolve=True)
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    pbar = tqdm(range(total_steps), disable=not accelerator.is_local_main_process)

    for epoch in range(args.epochs):
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        pbar.set_description(f"Epoch {epoch+1}")
        for batch in train_loader:
            with accelerator.accumulate(model):
                loss_data, frames = trainer.train_one_step(model, batch, step)
                loss = loss_data["loss_total"]

                optimizer.zero_grad()
                accelerator.backward(loss)
                optimizer.step()
                if accelerator.sync_gradients:
                    scheduler.step()

                meter.update("loss_total", accelerator.gather(loss).mean().item())
                for name, values in loss_data["log_data"].items():
                    meter.update(f"{name}", accelerator.gather(values).mean().item())

            if accelerator.sync_gradients:
                step += 1
                pbar.update(1)

                if step % args.log_every_steps == 0:
                    log_data = meter.average()
                    log_data["lr"] = optimizer.param_groups[0]["lr"]
                    accelerator.log(log_data, step=step)
                    pbar.set_postfix({k: f"{v:.3e}" if k == "lr" else f"{v:.3f}" for k, v in log_data.items()})
                    meter.reset()

                if step % args.visual_every_steps == 0:
                    log_train_visualization(
                        trainer,
                        frames,
                        step,
                        exp_dir,
                        accelerator,
                    )

                if step % args.val_every_steps == 0 or step == total_steps:
                    results = run_validation(model, val_loader, trainer, metrics, accelerator, step, exp_dir)
                    if accelerator.is_main_process and is_better(args.validate_metric, results[args.validate_metric], best_metric):
                        best_metric = results[args.validate_metric]
                        ckpt = {
                            "model": accelerator.unwrap_model(model).state_dict(),
                            **results,
                            "args": args,
                        }
                        torch.save(ckpt, f"{exp_dir}/checkpoints/best.pt")
                        logger.info(f"Saved the best {args.validate_metric}({best_metric:.4f}) checkpoints in {exp_dir}/checkpoints/best.pt")
                    accelerator.log({"best_val": best_metric, **results}, step=step)

    accelerator.end_training()


def main():
    args = load_config()
    accelerator = setup_accelerator(args)
    exp_dir = setup_experiment_dirs(args, accelerator)

    if args.global_seed is not None:
        set_seed(args.global_seed)

    train_loop(args, accelerator, exp_dir)
    logger.info("Training finished")


if __name__ == "__main__":
    main()
