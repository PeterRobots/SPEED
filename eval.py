import json
import argparse
import warnings
from omegaconf import OmegaConf
from collections import defaultdict

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from src.models import load_model
from src.datasets import load_dataset
from src.utils import CalMetrics, Trainer
from src.utils.setup_utils import setup_accelerator, setup_experiment_dirs

warnings.filterwarnings("ignore")
logger = get_logger(__name__, log_level="INFO")


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/eval_config.yaml")
    args = parser.parse_args()
    return OmegaConf.load(args.config)


@torch.no_grad()
def evaluation(args, accelerator, exp_dir):
    # load dataset
    local_batch_size = args.dataloader.batch_size
    dataset_name = args.dataset_name
    dataset = load_dataset(dataset_name, **args.dataset_args[dataset_name])
    dataloader = DataLoader(dataset, **args.dataloader)
    dataset_len = len(dataset)

    # load model
    model = load_model(args.model_name, **args.model_args)
    ckpt = torch.load(args.pretrained_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    logger.info(f"Model loaded from {args.pretrained_path}")
    del ckpt
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Full model params: {total_params:,}")

    model, dataloader = accelerator.prepare(model, dataloader)
    total_steps = len(dataloader)
    logger.info(f"Dataset {dataset_name} contains {dataset_len:,} triplets, local batch size is {local_batch_size}, evaluation needs {total_steps} steps")

    # begin evaluating
    model.eval()
    trainer = Trainer(None, device=accelerator.device)
    metrics = CalMetrics(accelerator.device)
    progress_bar = tqdm(
        range(0, total_steps),
        desc="Evaluating",
        disable=not accelerator.is_local_main_process,
    )

    results = defaultdict(list)
    for steps, batch in enumerate(dataloader):
        pred, gt = trainer.validate_one_step(model, batch)
        pred = (pred / 2 + 0.5).clamp(0, 1)
        gt = (gt / 2 + 0.5).clamp(0, 1)
        metrics_step = {
            "PSNR": metrics.cal_psnr(pred, gt),
            "SSIM": metrics.cal_ssim(pred, gt),
            "LPIPS": metrics.cal_lpips(pred, gt),
            "FloLPIPS": metrics.cal_flolpips(pred, gt, batch[:, 0] / 255., batch[:, 1] / 255.),
            "L1": torch.mean(torch.abs(pred - gt), dim=(1, 2, 3)),
        }
        for k, v in metrics_step.items():
            gathered = accelerator.gather_for_metrics(v)
            results[k].extend(gathered.cpu().tolist())
        progress_bar.update(1)
        if args.visualize:
            frame_0, frame_1 = batch[:, 0, ...] / 255., batch[:, 1, ...] / 255.
            save_image(frame_0, f"{exp_dir}/visualization_results/steps_{steps:04d}_frame_0_device_{accelerator.process_index}.png")
            save_image(frame_1, f"{exp_dir}/visualization_results/steps_{steps:04d}_frame_1_device_{accelerator.process_index}.png")
            save_image(gt, f"{exp_dir}/visualization_results/steps_{steps:04d}_gt_device_{accelerator.process_index}.png")
            save_image(pred, f"{exp_dir}/visualization_results/steps_{steps:04d}_pred_device_{accelerator.process_index}.png")
    final_results = {k: sum(v) / len(v) for k, v in results.items()}

    format_results = (
        f"PSNR: {final_results['PSNR']:.4f}, "
        f"SSIM: {final_results['SSIM']:.4f}, "
        f"LPIPS: {final_results['LPIPS']:.4f}, "
        f"FloLPIPS: {final_results['FloLPIPS']:.4f}, "
        f"L1: {final_results['L1']:.4f}"
    )
    logger.info(format_results)
    if accelerator.is_main_process:
        with open(f"{exp_dir}/eval_results.json", mode="w", encoding="utf-8") as f:
            json.dump(dict(results), f, indent=2, ensure_ascii=False)
    accelerator.end_training()
    model.eval()
    logger.info("Done!")


def main():
    args = load_config()
    accelerator = setup_accelerator(args, is_eval=True)
    exp_dir = setup_experiment_dirs(args, accelerator)

    if args.global_seed is not None:
        set_seed(args.global_seed)

    evaluation(args, accelerator, exp_dir)
    logger.info("Evaluation finished")


if __name__ == "__main__":
    main()
