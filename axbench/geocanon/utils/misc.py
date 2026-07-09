
from pathlib import Path
import os, torch

def safe_path_part(value):
    return str(value).replace(os.sep, "__").replace("/", "__")

def make_path(root, *parts):
    path=Path(root, *[safe_path_part(part) for part in parts])
    path.mkdir(parents=True, exist_ok=True)
    return path

def make_activation_dataset_dir_path(base_dir, dataset, dataset_config, split):
    return make_path(base_dir, dataset, dataset_config or "default", split)


def get_optimizer_cls(name):
    cls_dict = {
        "adam": torch.optim.Adam,
    }
    kwargs_dict = {}
    return cls_dict[name.lower()], kwargs_dict.get(name.lower(), {})

def get_model_args(name):
    kw=None
    if "nice" in name:
        parts=name.split("_")
        hidden_features=[]
        append_to_features=False
        for p in parts:
            if p=="hidden":
                append_to_features=True
            elif append_to_features:
                try:
                    hidden_features.append(int(p))
                except ValueError:
                    append_to_features=False

        kwargs_dict = {
            "hidden_features": tuple(hidden_features) if len(hidden_features)>0 else (128, 128),
            "context": 0,
            "randmask": True
        }
    return "nice", kwargs_dict