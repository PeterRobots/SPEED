from .model import SpeedDiT


def load_model(model_name, **model_args):
    if model_name != "SPEED":
        raise ValueError(f"No model named {model_name} in models!")
    return SpeedDiT(**model_args)
