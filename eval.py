import json
import argparse
import warnings

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from src.config import load_config as load_yaml_config
from src.datasets import load_dataset
from src.runtime import build_model_from_config
from src.utils import CalMetrics, Trainer
from src.utils.eval_utils import (
    average_metric_results,
    compute_metrics,
    denormalize_frame,
    format_metric_results,
    init_metric_results,
    update_metric_results,
)
from src.utils.setup_utils import setup_accelerator, setup_experiment_dirs

warnings.filterwarnings("ignore")
logger = get_logger(__name__, log_level="INFO")


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/eval_config.yaml")
    args = parser.parse_args()
    return load_yaml_config(args.config)


@torch.no_grad()
def evaluation(args, accelerator, exp_dir):
    # load dataset
    local_batch_size = args.dataloader.batch_size
    dataset_name = args.dataset_name
    dataset = load_dataset(dataset_name, **args.dataset_args[dataset_name])
    dataloader = DataLoader(dataset, **args.dataloader)
    dataset_len = len(dataset)

    # load model
    model, model_info = build_model_from_config(args, pretrained_path=args.pretrained_path, strict_load=True, require_checkpoint=True)
    logger.info(f"Model loaded from {model_info['checkpoint_path']}")
    logger.info(f"Full model params: {model_info['total_params']:,}")

    model, dataloader = accelerator.prepare(model, dataloader)
    total_steps = len(dataloader)
    logger.info(f"Dataset {dataset_name} contains {dataset_len:,} triplets, local batch size is {local_batch_size}, evaluation needs {total_steps} steps")

    # begin evaluating
    model.eval()
    trainer = Trainer(None, **args.trainer_args, device=accelerator.device)
    metrics = CalMetrics(accelerator.device)
    progress_bar = tqdm(
        range(0, total_steps),
        desc="Evaluating",
        disable=not accelerator.is_local_main_process,
    )

    results = init_metric_results()
    for steps, batch in enumerate(dataloader):
        pred, gt = trainer.validate_one_step(model, batch)
        pred = denormalize_frame(pred)
        gt = denormalize_frame(gt)
        metrics_step = compute_metrics(metrics, pred, gt, batch[:, 0] / 255., batch[:, 1] / 255.)
        update_metric_results(results, metrics_step, accelerator)
        progress_bar.update(1)
        if args.visualize:
            frame_0, frame_1 = batch[:, 0, ...] / 255., batch[:, 1, ...] / 255.
            save_image(frame_0, f"{exp_dir}/visualization_results/steps_{steps:04d}_frame_0_device_{accelerator.process_index}.png")
            save_image(frame_1, f"{exp_dir}/visualization_results/steps_{steps:04d}_frame_1_device_{accelerator.process_index}.png")
            save_image(gt, f"{exp_dir}/visualization_results/steps_{steps:04d}_gt_device_{accelerator.process_index}.png")
            save_image(pred, f"{exp_dir}/visualization_results/steps_{steps:04d}_pred_device_{accelerator.process_index}.png")
    logger.info(format_metric_results(average_metric_results(results)))
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
