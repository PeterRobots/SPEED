from importlib import import_module


__all__ = ["Trainer", "LossFunction", "CalMetrics", "FlowMatchScheduler"]

_EXPORT_MODULES = {
    "Trainer": ".trainer",
    "LossFunction": ".loss",
    "CalMetrics": ".metrics",
    "FlowMatchScheduler": ".scheduler",
}


def __getattr__(name):
    try:
        module_name = _EXPORT_MODULES[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value
