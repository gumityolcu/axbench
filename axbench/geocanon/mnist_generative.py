import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.utils import save_image

from generative.nf import load_flow_model_by_name, train_flow
from generative.vae import GaussianVAE, train_vae


IMAGE_SHAPE = (1, 28, 28)
INPUT_DIM = 28 * 28


class FlattenedImageDataset(Dataset):
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> torch.Tensor:
        image, _ = self.dataset[index]
        return image.reshape(-1)


def _device(device: str | torch.device | None) -> torch.device:
    if device is None or str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _dataset_cls(name: str):
    name = name.lower().replace("-", "_")
    if name == "mnist":
        return datasets.MNIST
    if name in {"fashion_mnist", "fashionmnist"}:
        return datasets.FashionMNIST
    raise ValueError(f"Unsupported dataset: {name!r}")


def make_mnist_datasets(
    dataset_name: str,
    data_dir: str | Path,
    download: bool = True,
) -> tuple[Dataset, Dataset]:
    dataset_cls = _dataset_cls(dataset_name)
    transform = transforms.ToTensor()

    train_data = FlattenedImageDataset(
        dataset_cls(
            root=str(data_dir),
            train=True,
            download=download,
            transform=transform,
        )
    )
    test_data = FlattenedImageDataset(
        dataset_cls(
            root=str(data_dir),
            train=False,
            download=download,
            transform=transform,
        )
    )

    return train_data, test_data


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


def _tuple_args(args: dict[str, Any], keys: Iterable[str]) -> dict[str, Any]:
    args = dict(args)
    for key in keys:
        if isinstance(args.get(key), list):
            args[key] = tuple(args[key])
    return args


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(data), f, indent=2)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_checkpoint(model_dir: Path, checkpoint: str | None) -> Path:
    if checkpoint in {None, "latest", "final"}:
        candidates = []
        for path in model_dir.glob("epoch*.pt"):
            epoch_text = path.stem.removeprefix("epoch")
            if epoch_text.isdigit():
                candidates.append((int(epoch_text), path))
        if not candidates:
            raise FileNotFoundError(f"No epoch*.pt checkpoints found in {model_dir}")
        return max(candidates, key=lambda item: item[0])[1]

    checkpoint_path = Path(checkpoint)
    if checkpoint_path.is_absolute() and checkpoint_path.exists():
        return checkpoint_path
    if checkpoint_path.exists():
        return checkpoint_path

    candidates = [model_dir / checkpoint]
    if checkpoint_path.suffix != ".pt":
        candidates.append(model_dir / f"{checkpoint}.pt")
        if checkpoint.isdigit():
            candidates.append(model_dir / f"epoch{checkpoint}.pt")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not resolve checkpoint {checkpoint!r} under {model_dir}"
    )


def _load_state_dict(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def default_nf_args() -> dict[str, Any]:
    return {
        "hidden_features": (512,512,512,512),
        "context": 0,
        "transforms": 3,
        "randmask": True,
    }


def default_vae_args(latent_dim: int = 64) -> dict[str, Any]:
    return {
        "latent_dim": latent_dim,
        "prior": "std",
        "encoder_hidden_features": (512, 512, 512, 512),
        "decoder_hidden_features": (512, 512, 512, 512),
    }


def train_mnist_nf_and_vae(
    save_dir: str | Path,
    dataset_name: str = "mnist",
    data_dir: str | Path = "data",
    epochs: int = 10,
    batch_size: int = 128,
    lr: float = 1e-4,
    save_freq: int = 0,
    device: str | torch.device | None = "auto",
    latent_dim: int = 64,
    train_nf_model: bool = True,
    train_vae_model: bool = True,
    download: bool = True,
    generate_checkpoints: bool = False,
    num_samples: int = 64,
    nrow: int = 8,
    seed: int | None = None,
) -> dict[str, Any]:
    device = _device(device)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_data, test_data = make_mnist_datasets(
        dataset_name=dataset_name,
        data_dir=data_dir,
        download=download,
    )

    metadata = {
        "dataset": dataset_name,
        "image_shape": IMAGE_SHAPE,
        "input_dim": INPUT_DIM,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "save_freq": save_freq,
        "nf": {
            "model_name": "nice",
            "model_args": default_nf_args(),
        },
        "vae": {
            "model_args": default_vae_args(latent_dim=latent_dim),
            "beta": 1.0,
        },
    }
    _write_json(save_dir / "mnist_generative_metadata.json", metadata)

    def make_generation_callback(model_name: str):
        if not generate_checkpoints:
            return None

        def generate_checkpoint_samples(checkpoint_name: str) -> None:
            cpu_rng_state = torch.get_rng_state()
            cuda_rng_states = (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None
            )
            try:
                outputs = generate_from_mnist_checkpoints(
                    save_dir=save_dir,
                    checkpoint=checkpoint_name,
                    num_samples=num_samples,
                    nrow=nrow,
                    device=device,
                    generate_nf_model=model_name == "nf",
                    generate_vae_model=model_name == "vae",
                    seed=seed,
                )
            finally:
                torch.set_rng_state(cpu_rng_state)
                if cuda_rng_states is not None:
                    torch.cuda.set_rng_state_all(cuda_rng_states)

            for output_model_name, path in outputs.items():
                print(f"{output_model_name}: {path}")

        return generate_checkpoint_samples

    if train_nf_model:
        nf_dir = save_dir / "nf"
        nf_dir.mkdir(parents=True, exist_ok=True)
        _write_json(nf_dir / "model_metadata.json", metadata["nf"])
        train_flow(
            train_data=train_data,
            test_data=test_data,
            model=metadata["nf"]["model_name"],
            optimizer_cls=torch.optim.Adam,
            optimizer_args={},
            lr=lr,
            batch_size=batch_size,
            epochs=epochs,
            device=device,
            model_args=metadata["nf"]["model_args"],
            save_freq=save_freq,
            save_dir=nf_dir,
            checkpoint_callback=make_generation_callback("nf"),
        )

    if train_vae_model:
        vae_dir = save_dir / "vae"
        vae_dir.mkdir(parents=True, exist_ok=True)
        _write_json(vae_dir / "model_metadata.json", metadata["vae"])
        train_vae(
            train_data=train_data,
            test_data=test_data,
            optimizer_cls=torch.optim.Adam,
            optimizer_args={},
            lr=lr,
            batch_size=batch_size,
            epochs=epochs,
            device=device,
            model_args=metadata["vae"]["model_args"],
            beta=metadata["vae"]["beta"],
            save_freq=save_freq,
            save_dir=vae_dir,
            checkpoint_callback=make_generation_callback("vae"),
        )

    return metadata


def _save_samples(samples: torch.Tensor, path: Path, nrow: int) -> Path:
    images = samples.detach().cpu().reshape(-1, *IMAGE_SHAPE).clamp(0.0, 1.0)
    save_image(images, path, nrow=nrow)
    return path


def generate_from_mnist_checkpoints(
    save_dir: str | Path,
    checkpoint: str | None = "latest",
    num_samples: int = 64,
    nrow: int = 8,
    device: str | torch.device | None = "auto",
    generate_nf_model: bool = True,
    generate_vae_model: bool = True,
    seed: int | None = None,
) -> dict[str, Path]:
    device = _device(device)
    save_dir = Path(save_dir)

    metadata_path = save_dir / "mnist_generative_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")
    metadata = _read_json(metadata_path)

    if tuple(metadata["image_shape"]) != IMAGE_SHAPE:
        raise ValueError(
            f"Expected image shape {IMAGE_SHAPE}, got {metadata['image_shape']}"
        )
    if metadata["input_dim"] != INPUT_DIM:
        raise ValueError(f"Expected input_dim {INPUT_DIM}, got {metadata['input_dim']}")

    if seed is not None:
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

    outputs: dict[str, Path] = {}

    if generate_nf_model:
        nf_dir = save_dir / "nf"
        nf_meta = metadata["nf"]
        nf_args = _tuple_args(nf_meta["model_args"], keys=("hidden_features",))
        nf_cls = load_flow_model_by_name(nf_meta["model_name"])
        nf = nf_cls(features=INPUT_DIM, **nf_args).to(device)
        nf_checkpoint = _resolve_checkpoint(nf_dir, checkpoint)
        nf.load_state_dict(_load_state_dict(nf_checkpoint, device))
        nf.eval()

        with torch.no_grad():
            nf_samples = nf().sample((num_samples,))

        outputs["nf"] = _save_samples(
            nf_samples,
            nf_checkpoint.parent / f"samples_{nf_checkpoint.stem}.png",
            nrow=nrow,
        )

    if generate_vae_model:
        vae_dir = save_dir / "vae"
        vae_meta = metadata["vae"]
        vae_args = _tuple_args(
            vae_meta["model_args"],
            keys=("encoder_hidden_features", "decoder_hidden_features"),
        )
        vae = GaussianVAE(input_dim=INPUT_DIM, beta=vae_meta["beta"], **vae_args).to(device)
        vae_checkpoint = _resolve_checkpoint(vae_dir, checkpoint)
        vae.load_state_dict(_load_state_dict(vae_checkpoint, device))
        vae.eval()

        with torch.no_grad():
            z = vae.prior().sample((num_samples,)).to(device)
            vae_samples = vae.decoder(z).mean

        outputs["vae"] = _save_samples(
            vae_samples,
            vae_checkpoint.parent / f"samples_{vae_checkpoint.stem}.png",
            nrow=nrow,
        )

    return outputs


def _parse_models(value: str) -> tuple[bool, bool]:
    value = value.lower()
    if value == "both":
        return True, True
    if value == "nf":
        return True, False
    if value == "vae":
        return False, True
    raise ValueError(f"Unsupported model selection: {value!r}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train NF/VAE models on MNIST-like datasets and sample from checkpoints."
    )
    parser.add_argument("--mode", choices=["train", "test", "both"], required=True)
    parser.add_argument("--dataset", choices=["mnist", "fashion_mnist"], default="mnist")
    parser.add_argument("--save_dir", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, default=Path("data"))
    parser.add_argument("--models", choices=["both", "nf", "vae"], default="both")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_freq", type=int, default=0)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--checkpoint", default="latest")
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--nrow", type=int, default=8)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no_download", action="store_true")
    args = parser.parse_args()

    use_nf, use_vae = _parse_models(args.models)

    if args.mode in {"train", "both"}:
        train_mnist_nf_and_vae(
            save_dir=args.save_dir,
            dataset_name=args.dataset,
            data_dir=args.data_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            save_freq=args.save_freq,
            device=args.device,
            latent_dim=args.latent_dim,
            train_nf_model=use_nf,
            train_vae_model=use_vae,
            download=not args.no_download,
            generate_checkpoints=args.mode == "both",
            num_samples=args.num_samples,
            nrow=args.nrow,
            seed=args.seed,
        )

    if args.mode == "test":
        outputs = generate_from_mnist_checkpoints(
            save_dir=args.save_dir,
            checkpoint=args.checkpoint,
            num_samples=args.num_samples,
            nrow=args.nrow,
            device=args.device,
            generate_nf_model=use_nf,
            generate_vae_model=use_vae,
            seed=args.seed,
        )
        for model_name, path in outputs.items():
            print(f"{model_name}: {path}")


if __name__ == "__main__":
    main()
