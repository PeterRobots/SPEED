from .model import SpeedDiT


MODELS = {
    "SPEED": SpeedDiT,
}


def load_model(model_name, **model_args):
    try:
        model_cls = MODELS[model_name]
    except KeyError as exc:
        raise ValueError(f"No model named {model_name} in models!") from exc
    return model_cls(**model_args)
