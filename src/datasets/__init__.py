from .LAVIB import LAVIBDataset
from .DAVIS import DAVISDataset
from .SNU_FILM import SNUFILMDataset
from .XTest import XTestDataset


def load_dataset(dataset_name, **dataset_args):
    if dataset_name == "LAVIB":
        return LAVIBDataset(**dataset_args)
    elif dataset_name == "DAVIS":
        return DAVISDataset(**dataset_args)
    elif dataset_name == "SNU_FILM":
        return SNUFILMDataset(**dataset_args)
    elif dataset_name == "XTest":
        return XTestDataset(**dataset_args)
    else:
        raise ValueError(f"No dataset named {dataset_name} in datasets!")
