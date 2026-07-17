import torch
from omegaconf import OmegaConf

from .models import load_model


def config_to_container(value):
    return OmegaConf.to_container(value, resolve=True) if value is not None else {}


def count_parameters(model):
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable


def load_checkpoint_state(path, map_location="cpu"):
    # SPEED training checkpoints contain an OmegaConf config in addition to the
    # model weights, so they are not compatible with PyTorch 2.6's default
    # weights_only=True loader. Only load checkpoints from trusted sources.
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint does not contain a valid state_dict: {path}")

    if state and all(key.startswith("module.") for key in state.keys()):
        state = {key[len("module."):]: value for key, value in state.items()}
    return state


def load_checkpoint(model, checkpoint_path, strict=False):
    state = load_checkpoint_state(checkpoint_path)
    missing, unexpected = model.load_state_dict(state, strict=strict)
    return missing, unexpected


def build_model_from_config(
    config,
    device=None,
    pretrained_path=None,
    strict_load=False,
    require_checkpoint=False,
    eval_mode=False,
):
    model_name = config.get("model_name", "SPEED")
    model_args = config_to_container(config.get("model_args", {}))
    model = load_model(model_name, **model_args)

    checkpoint_path = pretrained_path if pretrained_path is not None else config.get("pretrained_path")
    if require_checkpoint and not checkpoint_path:
        raise ValueError("Checkpoint path is required.")

    missing, unexpected = [], []
    if checkpoint_path:
        missing, unexpected = load_checkpoint(model, checkpoint_path, strict=strict_load)

    if device is not None:
        model.to(device)
    if eval_mode:
        model.eval()

    total_params, trainable_params = count_parameters(model)
    return model, {
        "model_name": model_name,
        "model_args": model_args,
        "checkpoint_path": checkpoint_path,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "total_params": total_params,
        "trainable_params": trainable_params,
    }
