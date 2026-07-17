import torch
from collections import defaultdict


METRIC_NAMES = ("PSNR", "SSIM", "LPIPS", "FloLPIPS", "L1")


def denormalize_frame(frame):
    return (frame / 2 + 0.5).clamp(0, 1)


def compute_metrics(metrics, pred, gt, frame0, frame1):
    return {
        "PSNR": metrics.cal_psnr(pred, gt),
        "SSIM": metrics.cal_ssim(pred, gt),
        "LPIPS": metrics.cal_lpips(pred, gt),
        "FloLPIPS": metrics.cal_flolpips(pred, gt, frame0, frame1),
        "L1": torch.mean(torch.abs(pred - gt), dim=(1, 2, 3)),
    }


def update_metric_results(results, metrics_step, accelerator):
    for name, values in metrics_step.items():
        gathered = accelerator.gather_for_metrics(values)
        results[name].extend(gathered.cpu().tolist())


def init_metric_results():
    return defaultdict(list)


def average_metric_results(results):
    empty_metrics = [name for name, values in results.items() if not values]
    if not results or empty_metrics:
        raise ValueError(f"No metric values were collected: {empty_metrics or list(results.keys())}")
    return {name: sum(values) / len(values) for name, values in results.items()}


def format_metric_results(results):
    return ", ".join(f"{name}: {results[name]:.4f}" for name in METRIC_NAMES if name in results)
