from .LAVIB import LAVIBDataset
from .DAVIS import DAVISDataset
from .SNU_FILM import SNUFILMDataset
from .XTest import XTestDataset


DATASETS = {
    "LAVIB": LAVIBDataset,
    "DAVIS": DAVISDataset,
    "SNU_FILM": SNUFILMDataset,
    "XTest": XTestDataset,
}


def load_dataset(dataset_name, **dataset_args):
    try:
        dataset_cls = DATASETS[dataset_name]
    except KeyError as exc:
        raise ValueError(f"No dataset named {dataset_name} in datasets!") from exc
    return dataset_cls(**dataset_args)
