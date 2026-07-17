from pathlib import Path

from omegaconf import OmegaConf


def load_config(config_path):
    config_path = Path(config_path).expanduser()
    config = OmegaConf.load(config_path)
    defaults = config.get("defaults", [])

    if defaults is None:
        defaults = []
    if isinstance(defaults, str):
        defaults = [defaults]

    merged = OmegaConf.create()
    for default in defaults:
        if not isinstance(default, str):
            raise TypeError(f"Unsupported config default entry: {default!r}")
        default_path = Path(default).expanduser()
        if not default_path.is_absolute():
            default_path = config_path.parent / default_path
        merged = OmegaConf.merge(merged, load_config(default_path))

    if "defaults" in config:
        del config["defaults"]
    return OmegaConf.merge(merged, config)
