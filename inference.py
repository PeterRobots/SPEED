import argparse
from contextlib import nullcontext
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Run SPEED frame interpolation on image pairs or videos.")
    parser.add_argument("--config", type=str, default="configs/eval_config.yaml", help="Path to model config.")
    parser.add_argument("--pretrained_path", type=str, default=None, help="Checkpoint path. Overrides config.pretrained_path.")

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input_video", type=str, default=None, help="Input video path.")
    input_group.add_argument("--frame0", type=str, default=None, help="First input image path.")
    parser.add_argument("--frame1", type=str, default=None, help="Second input image path. Required when --frame0 is used.")

    parser.add_argument("--output", type=str, required=True, help="Output image or video path.")
    parser.add_argument("--video_mode", type=str, choices=("parallel", "sequential"), default="parallel",
                        help="Video interpolation mode. parallel uses batching; sequential uses one pair at a time.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for video parallel mode.")
    parser.add_argument("--keep_fps", action="store_true",
                        help="Keep input FPS. By default output FPS is doubled to preserve playback duration.")
    parser.add_argument("--fps", type=float, default=None, help="Override output FPS.")
    parser.add_argument("--fourcc", type=str, default="mp4v", help="OpenCV fourcc for output video.")

    parser.add_argument("--device", type=str, default=None, help="Device, e.g. cuda:0 or cpu. Defaults to CUDA if available.")
    parser.add_argument("--precision", type=str, choices=("fp32", "fp16", "bf16"), default="fp32",
                        help="Autocast precision used on CUDA.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for the inference noise input.")
    parser.add_argument("--strict_load", action="store_true", help="Load checkpoint with strict=True.")
    return parser.parse_args()


def ensure_parent_dir(path):
    parent = Path(path).expanduser().resolve().parent
    parent.mkdir(parents=True, exist_ok=True)


def require_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("PyTorch is required for inference. Install torch and torchvision first.") from exc
    return torch


def require_cv2():
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("opencv-python is required for inference. Install it with: pip install opencv-python") from exc
    return cv2


class NullProgressBar:
    def update(self, _=1):
        pass

    def close(self):
        pass


def create_progress_bar(total, desc):
    try:
        from tqdm import tqdm
    except ModuleNotFoundError:
        return NullProgressBar()
    return tqdm(total=total, desc=desc)


def read_rgb_image(path):
    cv2_module = require_cv2()
    image = cv2_module.imread(str(path), cv2_module.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return cv2_module.cvtColor(image, cv2_module.COLOR_BGR2RGB)


def write_rgb_image(path, image):
    cv2_module = require_cv2()
    ensure_parent_dir(path)
    suffix = Path(path).suffix.lower()
    if suffix not in IMAGE_SUFFIXES:
        raise ValueError(f"Output image suffix must be one of {sorted(IMAGE_SUFFIXES)}, got: {suffix}")
    image = cv2_module.cvtColor(image, cv2_module.COLOR_RGB2BGR)
    ok = cv2_module.imwrite(str(path), image)
    if not ok:
        raise RuntimeError(f"Failed to write image: {path}")


def rgb_to_tensor(image):
    torch = require_torch()
    tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous()
    return tensor.float()


def tensor_to_rgb(tensor):
    tensor = tensor.detach().float().clamp(0, 1)
    tensor = (tensor * 255.0).round().byte().cpu()
    return tensor.permute(1, 2, 0).numpy()


def normalize_frames(frames):
    return frames / 127.5 - 1.0


def denormalize_pred(pred):
    return (pred / 2.0 + 0.5).clamp(0, 1)


def get_device(device_arg):
    torch = require_torch()
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_autocast_dtype(precision):
    torch = require_torch()
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    return None


def load_checkpoint_state(path):
    torch = require_torch()
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint does not contain a valid state_dict: {path}")

    if state and all(key.startswith("module.") for key in state.keys()):
        state = {key[len("module."):]: value for key, value in state.items()}
    return state


def build_model(config_path, pretrained_path, device, strict_load=False):
    from omegaconf import OmegaConf
    from src.models import load_model

    args = OmegaConf.load(config_path)
    ckpt_path = pretrained_path or args.get("pretrained_path")
    if not ckpt_path:
        raise ValueError("Checkpoint path is required. Set --pretrained_path or config.pretrained_path.")

    model_name = args.get("model_name", "SPEED")
    model_args = args.get("model_args", {})
    model = load_model(model_name, **model_args)

    state = load_checkpoint_state(ckpt_path)
    missing, unexpected = model.load_state_dict(state, strict=strict_load)
    if missing:
        print(f"[WARN] Missing checkpoint keys: {len(missing)}")
    if unexpected:
        print(f"[WARN] Unexpected checkpoint keys: {len(unexpected)}")

    model.to(device)
    model.eval()
    return model


def interpolate_batch(model, frame0_list, frame1_list, device, precision="fp32"):
    torch = require_torch()
    if len(frame0_list) != len(frame1_list):
        raise ValueError("frame0_list and frame1_list must have the same length.")
    if not frame0_list:
        return []

    shapes = {frame.shape for frame in frame0_list + frame1_list}
    if len(shapes) != 1:
        raise ValueError(f"All frames in one batch must have the same shape, got: {sorted(shapes)}")

    frame0 = torch.stack([rgb_to_tensor(frame) for frame in frame0_list], dim=0).to(device, non_blocking=True)
    frame1 = torch.stack([rgb_to_tensor(frame) for frame in frame1_list], dim=0).to(device, non_blocking=True)

    frame0 = normalize_frames(frame0)
    frame1 = normalize_frames(frame1)
    cond_frames = torch.cat((frame0, frame1), dim=0)

    noisy_frames = torch.randn_like(frame0)
    timestep = torch.full((frame0.shape[0],), 1000.0, dtype=frame0.dtype, device=device)

    autocast_dtype = get_autocast_dtype(precision)
    autocast_enabled = device.type == "cuda" and autocast_dtype is not None
    autocast_context = (
        torch.autocast(device_type="cuda", dtype=autocast_dtype)
        if autocast_enabled
        else nullcontext()
    )
    with torch.no_grad(), autocast_context:
        pred = model(noisy_frames=noisy_frames, cond_frames=cond_frames, timestep=timestep)

    pred = denormalize_pred(pred)
    return [tensor_to_rgb(pred[i]) for i in range(pred.shape[0])]


def interpolate_image_pair(model, frame0_path, frame1_path, output_path, device, precision):
    frame0 = read_rgb_image(frame0_path)
    frame1 = read_rgb_image(frame1_path)
    if frame0.shape != frame1.shape:
        raise ValueError(f"Input images must have the same shape, got {frame0.shape} and {frame1.shape}.")

    pred = interpolate_batch(model, [frame0], [frame1], device, precision=precision)[0]
    write_rgb_image(output_path, pred)
    print(f"Saved interpolated image to: {output_path}")


def open_video_reader(path):
    cv2_module = require_cv2()
    cap = cv2_module.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video: {path}")
    return cap


def read_next_rgb_frame(cap):
    cv2_module = require_cv2()
    ok, frame = cap.read()
    if not ok:
        return None
    return cv2_module.cvtColor(frame, cv2_module.COLOR_BGR2RGB)


def create_video_writer(path, fps, width, height, fourcc):
    cv2_module = require_cv2()
    ensure_parent_dir(path)
    writer = cv2_module.VideoWriter(
        str(path),
        cv2_module.VideoWriter_fourcc(*fourcc),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create output video: {path}")
    return writer


def write_rgb_video_frame(writer, frame):
    cv2_module = require_cv2()
    writer.write(cv2_module.cvtColor(frame, cv2_module.COLOR_RGB2BGR))


def resolve_output_fps(input_fps, fps_override, keep_fps):
    if fps_override is not None:
        return fps_override
    if keep_fps:
        return input_fps
    return input_fps * 2.0


def flush_video_batch(model, writer, left_frames, right_frames, device, precision):
    mids = interpolate_batch(model, left_frames, right_frames, device, precision=precision)
    for left, mid in zip(left_frames, mids):
        write_rgb_video_frame(writer, left)
        write_rgb_video_frame(writer, mid)


def interpolate_video_parallel(model, input_path, output_path, device, precision, batch_size, keep_fps, fps_override, fourcc):
    if batch_size <= 0:
        raise ValueError("--batch_size must be greater than 0.")

    cap = open_video_reader(input_path)
    cv2_module = require_cv2()
    input_fps = cap.get(cv2_module.CAP_PROP_FPS)
    if input_fps <= 0:
        input_fps = 25.0

    frame_count = int(cap.get(cv2_module.CAP_PROP_FRAME_COUNT))
    first = read_next_rgb_frame(cap)
    if first is None:
        cap.release()
        raise ValueError(f"Input video has no frames: {input_path}")

    height, width = first.shape[:2]
    output_fps = resolve_output_fps(input_fps, fps_override, keep_fps)
    writer = create_video_writer(output_path, output_fps, width, height, fourcc)

    left_frames = []
    right_frames = []
    prev = first
    total_pairs = max(frame_count - 1, 0)
    pbar = create_progress_bar(total=total_pairs, desc="Interpolating video pairs")

    try:
        while True:
            cur = read_next_rgb_frame(cap)
            if cur is None:
                break
            if cur.shape != prev.shape:
                raise ValueError(f"Video frame shape changed from {prev.shape} to {cur.shape}.")

            left_frames.append(prev)
            right_frames.append(cur)
            prev = cur

            if len(left_frames) >= batch_size:
                flush_video_batch(model, writer, left_frames, right_frames, device, precision)
                pbar.update(len(left_frames))
                left_frames.clear()
                right_frames.clear()

        if left_frames:
            flush_video_batch(model, writer, left_frames, right_frames, device, precision)
            pbar.update(len(left_frames))

        write_rgb_video_frame(writer, prev)
    finally:
        pbar.close()
        cap.release()
        writer.release()

    print(f"Saved interpolated video to: {output_path}")


def interpolate_video_sequential(model, input_path, output_path, device, precision, keep_fps, fps_override, fourcc):
    cap = open_video_reader(input_path)
    cv2_module = require_cv2()
    input_fps = cap.get(cv2_module.CAP_PROP_FPS)
    if input_fps <= 0:
        input_fps = 25.0

    frame_count = int(cap.get(cv2_module.CAP_PROP_FRAME_COUNT))
    first = read_next_rgb_frame(cap)
    if first is None:
        cap.release()
        raise ValueError(f"Input video has no frames: {input_path}")

    height, width = first.shape[:2]
    output_fps = resolve_output_fps(input_fps, fps_override, keep_fps)
    writer = create_video_writer(output_path, output_fps, width, height, fourcc)

    prev = first
    total_pairs = max(frame_count - 1, 0)
    pbar = create_progress_bar(total=total_pairs, desc="Interpolating video pairs")

    try:
        while True:
            cur = read_next_rgb_frame(cap)
            if cur is None:
                break
            if cur.shape != prev.shape:
                raise ValueError(f"Video frame shape changed from {prev.shape} to {cur.shape}.")

            mid = interpolate_batch(model, [prev], [cur], device, precision=precision)[0]
            write_rgb_video_frame(writer, prev)
            write_rgb_video_frame(writer, mid)
            prev = cur
            pbar.update(1)

        write_rgb_video_frame(writer, prev)
    finally:
        pbar.close()
        cap.release()
        writer.release()

    print(f"Saved interpolated video to: {output_path}")


def validate_args(args):
    if args.frame0 and not args.frame1:
        raise ValueError("--frame1 is required when --frame0 is used.")
    if args.input_video and args.frame1:
        raise ValueError("--frame1 can only be used together with --frame0.")
    if len(args.fourcc) != 4:
        raise ValueError("--fourcc must be exactly 4 characters, e.g. mp4v.")


def main():
    args = parse_args()
    validate_args(args)

    torch = require_torch()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = get_device(args.device)
    if device.type == "cpu":
        print("[WARN] Running on CPU. SPEED uses xFormers attention and is intended for CUDA inference.")

    model = build_model(args.config, args.pretrained_path, device, strict_load=args.strict_load)

    if args.frame0:
        interpolate_image_pair(model, args.frame0, args.frame1, args.output, device, args.precision)
        return

    if args.video_mode == "parallel":
        interpolate_video_parallel(
            model,
            args.input_video,
            args.output,
            device,
            args.precision,
            args.batch_size,
            args.keep_fps,
            args.fps,
            args.fourcc,
        )
    else:
        interpolate_video_sequential(
            model,
            args.input_video,
            args.output,
            device,
            args.precision,
            args.keep_fps,
            args.fps,
            args.fourcc,
        )


if __name__ == "__main__":
    main()
