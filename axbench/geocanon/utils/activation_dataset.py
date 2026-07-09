import json
from pathlib import Path

import h5py
import torch
from torch.utils.data import Dataset

from utils.misc import make_path,  make_activation_dataset_dir_path

class ActivationDataset(Dataset):
    """Torch dataset for activation rows written by extract_features.py.

    The HDF5 dataset is written append-only in token order. This dataset keeps
    that order exactly: item i is row i from the HDF5 dataset.
    """

    def __init__(
        self,
        model_name,
        layer,
        dataset,
        split,
        device,
        base_dir="/mnt/storage/yolcu/geocanon-activations",
        dataset_config=None,
        dtype=torch.bfloat16,
    ):
        dataset_dir = make_activation_dataset_dir_path(base_dir, dataset, dataset_config, split)
        self.token_manifest_path = dataset_dir / "token_manifest.jsonl"
        dataset_dir = make_path(dataset_dir, model_name, layer)
        self.manifest_path = dataset_dir / "manifest.jsonl"
        self.h5_path = dataset_dir / "activations.hdf5"

        manifest = self.read_manifest(self.manifest_path)

        h5_path = manifest.get("path", None)
        if h5_path is None:
            raise ValueError(f"`path` is missing in manifest: {self.manifest_path}")

        self.h5_path = Path(h5_path)
        if not self.h5_path.is_absolute() and self.manifest_path is not None:
            self.h5_path = self.manifest_path.parent / self.h5_path

        self.hdf5_dataset = manifest.get("hdf5_dataset")
        self.expected_rows = manifest.get("n_rows")
        self.layer = manifest.get("layer")
        assert self.layer == layer, f"Manifest layer {self.layer} does not match expected {layer}"
        self.dtype = dtype
        self.device = device

        self._h5_file = None
        self._activations = None
        self._length = self._read_length()

        if self.expected_rows is not None and self.expected_rows != self._length:
            raise ValueError(
                f"Manifest n_rows={self.expected_rows} but "
                f"{self.h5_path}:{self.hdf5_dataset} has {self._length} rows"
            )

    @staticmethod
    def read_manifest(manifest_path):
        if manifest_path is None:
            return {}

        with manifest_path.open("r", encoding="utf-8") as manifest_file:
            line = manifest_file.readline()

        if not line:
            raise ValueError(f"Manifest is empty: {manifest_path}")
        return json.loads(line)

    def _ensure_open(self):
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, "r")
            self._activations = self._h5_file[self.hdf5_dataset]
        return self._activations

    def _read_length(self):
        with h5py.File(self.h5_path, "r") as h5_file:
            return h5_file[self.hdf5_dataset].shape[0]

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        if isinstance(index, slice):
            raise TypeError("ActivationDataset expects integer indices")
        if index < 0:
            index += self._length
        if index < 0 or index >= self._length:
            raise IndexError(index)

        row = self._ensure_open()[index]
        tensor = torch.from_numpy(row)
        if self.dtype is not None:
            tensor = tensor.to(self.dtype)
        return tensor

    def close(self):
        if self._h5_file is not None:
            self._h5_file.close()
            self._h5_file = None
            self._activations = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5_file"] = None
        state["_activations"] = None
        return state

    def __del__(self):
        self.close()
