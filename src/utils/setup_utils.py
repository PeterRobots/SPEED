import os
import logging
import requests
from datetime import datetime

import diffusers
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import broadcast_object_list
from swanlab.integration.accelerate import SwanLabTracker

logger = get_logger(__name__, log_level="INFO")


def is_http_reachable(url="https://swanlab.cn/", timeout=3):
    try:
        response = requests.get(url, timeout=timeout)
        return response.status_code == 200
    except requests.RequestException:
        return False


def setup_accelerator(args, is_eval=False):
    if not is_eval:
        requested_mode = os.environ.get("SWANLAB_MODE")
        tracker_mode = requested_mode or (None if is_http_reachable() else "local")
        tracker = SwanLabTracker(
            project=args.tracker_project_name,
            experiment_name=args.experiment_name,
            mode=tracker_mode,
        )
        accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision=args.mixed_precision,
            log_with=[tracker],
        )
    else:
        accelerator = Accelerator(
            mixed_precision=args.mixed_precision
        )
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
    return accelerator


def setup_experiment_dirs(args, accelerator):
    output_root = f"{args.output_dir}/{args.experiment_name}"
    if accelerator.is_main_process:
        os.makedirs(output_root, exist_ok=True)
        exp_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        exp_id = None
    exp_id = broadcast_object_list([exp_id])[0]
    exp_dir = f"{output_root}/{exp_id}"

    if accelerator.is_main_process:
        os.makedirs(exp_dir, exist_ok=True)
        for name in ["checkpoints", "visualization_results", "validation_results"]:
            os.makedirs(f"{exp_dir}/{name}", exist_ok=True)
        logging.basicConfig(
            format="[\033[34m%(asctime)s\033[0m]-%(message)s",
            datefmt="%Y/%m/%d %H:%M:%S",
            level=logging.INFO,
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{exp_dir}/log.txt")],
        )
        logger.info(f"Experiment dir: {exp_dir}")
    accelerator.wait_for_everyone()
    return exp_dir
