import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path


torch = None


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark SPEED inference time and CUDA memory on random inputs.")
    parser.add_argument("--config", type=str, default="configs/eval_config.yaml", help="Path to model config.")
    parser.add_argument("--pretrained_path", type=str, default=None,
                        help="Optional checkpoint path. If omitted, benchmark a randomly initialized model.")
    parser.add_argument("--height", type=int, required=True, help="Input height.")
    parser.add_argument("--width", type=int, required=True, help="Input width.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size of predicted middle frames.")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup forward passes.")
    parser.add_argument("--runs", type=int, default=100, help="Measured forward passes.")
    parser.add_argument("--device", type=str, default=None, help="Device, e.g. cuda:0 or cpu. Defaults to CUDA if available.")
    parser.add_argument("--precision", type=str, choices=("fp32", "fp16", "bf16"), default="fp32",
                        help="Autocast precision used on CUDA.")
    parser.add_argument("--timestep", type=float, default=1000.0, help="Inference timestep value.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed used for model/input initialization.")
    parser.add_argument("--strict_load", action="store_true", help="Load checkpoint with strict=True.")
    parser.add_argument("--empty_cache", action="store_true", help="Call torch.cuda.empty_cache() before benchmarking.")
    parser.add_argument("--output_json", type=str, default=None, help="Optional path to save benchmark results as JSON.")
    return parser.parse_args()


def require_torch():
    global torch
    if torch is None:
        import torch as torch_module
        torch = torch_module
    return torch


def load_config(config_path):
    try:
        from src.config import load_config as load_yaml_config
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("omegaconf is required to read SPEED config files.") from exc
    return load_yaml_config(config_path)


def get_device(device_arg):
    torch_module = require_torch()
    if device_arg:
        device = torch_module.device(device_arg)
        if device.type == "cuda" and not torch_module.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but CUDA is not available: {device_arg}")
        return device
    return torch_module.device("cuda:0" if torch_module.cuda.is_available() else "cpu")


def get_autocast_context(device, precision):
    torch_module = require_torch()
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch_module.float16 if precision == "fp16" else torch_module.bfloat16
    return torch_module.autocast(device_type="cuda", dtype=dtype)


def build_model(config_path, pretrained_path, device, strict_load):
    from src.runtime import build_model_from_config

    config = load_config(config_path)
    checkpoint_path = pretrained_path or ""
    model, model_info = build_model_from_config(
        config,
        device=device,
        pretrained_path=checkpoint_path,
        strict_load=strict_load,
        eval_mode=True,
    )
    if model_info["checkpoint_path"]:
        if model_info["missing_keys"]:
            print(f"[WARN] Missing checkpoint keys: {len(model_info['missing_keys'])}")
        if model_info["unexpected_keys"]:
            print(f"[WARN] Unexpected checkpoint keys: {len(model_info['unexpected_keys'])}")
        print(f"Loaded checkpoint: {model_info['checkpoint_path']}")
    else:
        print("No checkpoint is provided. Benchmarking a randomly initialized model.")

    return model, model_info["model_name"], model_info["model_args"]


def build_random_inputs(batch_size, height, width, timestep, device):
    torch_module = require_torch()
    noisy_frames = torch_module.randn(batch_size, 3, height, width, device=device)
    cond_frames = torch_module.randn(batch_size * 2, 3, height, width, device=device)
    timestep = torch_module.full((batch_size,), float(timestep), dtype=noisy_frames.dtype, device=device)
    return {
        "noisy_frames": noisy_frames,
        "cond_frames": cond_frames,
        "timestep": timestep,
    }


def bytes_to_mib(value):
    return value / 1024.0 / 1024.0


def summarize(values):
    values = sorted(values)
    count = len(values)
    mean = sum(values) / count
    median = values[count // 2] if count % 2 else (values[count // 2 - 1] + values[count // 2]) / 2.0
    variance = sum((value - mean) ** 2 for value in values) / count
    return {
        "mean": mean,
        "median": median,
        "min": values[0],
        "max": values[-1],
        "std": variance ** 0.5,
    }


def cuda_memory_snapshot(device):
    torch_module = require_torch()
    return {
        "allocated_mib": bytes_to_mib(torch_module.cuda.memory_allocated(device)),
        "reserved_mib": bytes_to_mib(torch_module.cuda.memory_reserved(device)),
        "max_allocated_mib": bytes_to_mib(torch_module.cuda.max_memory_allocated(device)),
        "max_reserved_mib": bytes_to_mib(torch_module.cuda.max_memory_reserved(device)),
    }


def run_forward(model, inputs, device, precision):
    torch_module = require_torch()
    with torch_module.inference_mode():
        with get_autocast_context(device, precision):
            return model(**inputs)


def benchmark_cuda(model, inputs, device, precision, warmup, runs, empty_cache):
    torch_module = require_torch()
    output = None
    for _ in range(warmup):
        output = run_forward(model, inputs, device, precision)
    del output
    torch_module.cuda.synchronize(device)

    if empty_cache:
        torch_module.cuda.empty_cache()

    baseline_allocated = torch_module.cuda.memory_allocated(device)
    baseline_reserved = torch_module.cuda.memory_reserved(device)

    times_ms = []
    peak_allocated_mib = []
    peak_reserved_mib = []
    extra_allocated_mib = []
    extra_reserved_mib = []

    start_event = torch_module.cuda.Event(enable_timing=True)
    end_event = torch_module.cuda.Event(enable_timing=True)

    output = None
    for _ in range(runs):
        output = None
        torch_module.cuda.synchronize(device)
        torch_module.cuda.reset_peak_memory_stats(device)

        start_event.record()
        output = run_forward(model, inputs, device, precision)
        end_event.record()
        torch_module.cuda.synchronize(device)

        elapsed_ms = start_event.elapsed_time(end_event)
        peak_allocated = torch_module.cuda.max_memory_allocated(device)
        peak_reserved = torch_module.cuda.max_memory_reserved(device)

        times_ms.append(elapsed_ms)
        peak_allocated_mib.append(bytes_to_mib(peak_allocated))
        peak_reserved_mib.append(bytes_to_mib(peak_reserved))
        extra_allocated_mib.append(bytes_to_mib(max(0, peak_allocated - baseline_allocated)))
        extra_reserved_mib.append(bytes_to_mib(max(0, peak_reserved - baseline_reserved)))

        del output

    torch_module.cuda.synchronize(device)
    return {
        "time_ms": summarize(times_ms),
        "throughput_fps": inputs["noisy_frames"].shape[0] / (sum(times_ms) / len(times_ms) / 1000.0),
        "memory": {
            "baseline_allocated_mib": bytes_to_mib(baseline_allocated),
            "baseline_reserved_mib": bytes_to_mib(baseline_reserved),
            "peak_allocated_mib": summarize(peak_allocated_mib),
            "peak_reserved_mib": summarize(peak_reserved_mib),
            "forward_extra_allocated_mib": summarize(extra_allocated_mib),
            "forward_extra_reserved_mib": summarize(extra_reserved_mib),
            "final": cuda_memory_snapshot(device),
        },
    }


def benchmark_cpu(model, inputs, device, precision, warmup, runs):
    output = None
    for _ in range(warmup):
        output = run_forward(model, inputs, device, precision)
    del output

    times_ms = []
    for _ in range(runs):
        start = time.perf_counter()
        output = run_forward(model, inputs, device, precision)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        times_ms.append(elapsed_ms)
        del output

    return {
        "time_ms": summarize(times_ms),
        "throughput_fps": inputs["noisy_frames"].shape[0] / (sum(times_ms) / len(times_ms) / 1000.0),
        "memory": None,
    }


def print_results(results):
    time_stats = results["time_ms"]
    print("\nTime")
    print(f"  mean   : {time_stats['mean']:.3f} ms")
    print(f"  median : {time_stats['median']:.3f} ms")
    print(f"  min/max: {time_stats['min']:.3f} / {time_stats['max']:.3f} ms")
    print(f"  std    : {time_stats['std']:.3f} ms")
    print(f"  throughput: {results['throughput_fps']:.3f} frames/s")

    memory = results["memory"]
    if memory is None:
        print("\nMemory")
        print("  CUDA memory is unavailable on CPU.")
        return

    print("\nCUDA Memory")
    print(f"  baseline allocated: {memory['baseline_allocated_mib']:.2f} MiB")
    print(f"  baseline reserved : {memory['baseline_reserved_mib']:.2f} MiB")
    print(f"  peak allocated    : {memory['peak_allocated_mib']['max']:.2f} MiB")
    print(f"  peak reserved     : {memory['peak_reserved_mib']['max']:.2f} MiB")
    print(f"  forward extra alloc max: {memory['forward_extra_allocated_mib']['max']:.2f} MiB")
    print(f"  forward extra rsvd max : {memory['forward_extra_reserved_mib']['max']:.2f} MiB")


def main():
    args = parse_args()
    if args.height <= 0 or args.width <= 0:
        raise ValueError("--height and --width must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative.")
    if args.runs <= 0:
        raise ValueError("--runs must be positive.")

    torch_module = require_torch()
    torch_module.manual_seed(args.seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(args.seed)

    device = get_device(args.device)
    if device.type == "cuda":
        torch_module.cuda.set_device(device)
    model, model_name, model_args = build_model(args.config, args.pretrained_path, device, args.strict_load)
    inputs = build_random_inputs(args.batch_size, args.height, args.width, args.timestep, device)

    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

    print("Benchmark Config")
    print(f"  model      : {model_name}")
    print(f"  model_args : {model_args}")
    print(f"  params     : {total_params:,} total / {trainable_params:,} trainable")
    print(f"  input      : batch={args.batch_size}, height={args.height}, width={args.width}")
    print(f"  cond_frames: {tuple(inputs['cond_frames'].shape)}")
    print(f"  noisy      : {tuple(inputs['noisy_frames'].shape)}")
    print(f"  device     : {device}")
    print(f"  precision  : {args.precision}")
    print(f"  warmup/runs: {args.warmup}/{args.runs}")

    if device.type == "cuda":
        results = benchmark_cuda(model, inputs, device, args.precision, args.warmup, args.runs, args.empty_cache)
    else:
        results = benchmark_cpu(model, inputs, device, args.precision, args.warmup, args.runs)

    results.update({
        "model": model_name,
        "model_args": model_args,
        "params": {
            "total": total_params,
            "trainable": trainable_params,
        },
        "input": {
            "batch_size": args.batch_size,
            "height": args.height,
            "width": args.width,
            "timestep": args.timestep,
        },
        "device": str(device),
        "precision": args.precision,
        "warmup": args.warmup,
        "runs": args.runs,
    })

    print_results(results)

    if args.output_json:
        output_path = Path(args.output_json).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nSaved benchmark JSON to: {output_path}")


if __name__ == "__main__":
    main()
