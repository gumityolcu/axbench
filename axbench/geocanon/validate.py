import argparse
import json
import re
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from generative.nf import load_flow_model_by_name
from utils.activation_dataset import ActivationDataset
from utils.misc import get_model_args, make_activation_dataset_dir_path, make_path


class MomentAccumulator:
    def __init__(self, features, device="cpu", dtype=torch.float64):
        self.n = 0
        self.sum = torch.zeros(features, device=device, dtype=dtype)
        self.sum_outer = torch.zeros(features, features, device=device, dtype=dtype)

    def update(self, batch):
        if batch.ndim != 2:
            raise ValueError(f"Expected a 2D activation batch, got shape {tuple(batch.shape)}")
        batch = batch.to(device=self.sum.device, dtype=self.sum.dtype)
        self.n += batch.shape[0]
        self.sum += batch.sum(dim=0)
        self.sum_outer += batch.T @ batch

    def mean_and_cov(self):
        if self.n < 2:
            raise ValueError("Need at least two samples to estimate covariance")

        mean = self.sum / self.n
        cov = (self.sum_outer - self.n * torch.outer(mean, mean)) / (self.n - 1)
        cov = (cov + cov.T) * 0.5
        return mean, cov


def _checkpoint_epoch(path):
    match = re.fullmatch(r"epoch(\d+)\.pt", path.name)
    return int(match.group(1)) if match else None


def resolve_checkpoint(save_dir, checkpoint):
    save_dir = Path(save_dir)

    if checkpoint in {"final", "latest"}:
        final_path = save_dir / "final.pt"
        if final_path.exists():
            return final_path

        candidates = []
        for path in save_dir.glob("epoch*.pt"):
            epoch = _checkpoint_epoch(path)
            if epoch is not None:
                candidates.append((epoch, path))

        if not candidates:
            raise FileNotFoundError(
                f"No epoch*.pt checkpoints found in {save_dir}; "
                "pass --checkpoint with a checkpoint path or filename."
            )
        return max(candidates, key=lambda item: item[0])[1]

    path = Path(checkpoint)
    if path.is_absolute() and path.exists():
        return path
    if path.exists():
        return path

    candidates = [save_dir / checkpoint]
    if path.suffix != ".pt":
        candidates.append(save_dir / f"{checkpoint}.pt")
        if checkpoint.isdigit():
            candidates.append(save_dir / f"epoch{checkpoint}.pt")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not resolve checkpoint {checkpoint!r}. Tried: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def load_generative_model(gen_model_name, gen_model_type, features, checkpoint_path, device):
    model_name, model_args = get_model_args(gen_model_name)

    if gen_model_type == "nf":
        flow_cls = load_flow_model_by_name(model_name)
        model = flow_cls(features=features, **model_args)
    else:
        raise ValueError(f"Unsupported model type: {gen_model_type}")

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def accumulate_real_moments(dataset, num_samples, batch_size, stats_device):
    accumulator = MomentAccumulator(
        features=dataset[0].shape[0],
        device=stats_device,
        dtype=torch.float64,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for batch in loader:
        remaining = num_samples - accumulator.n
        if remaining <= 0:
            break
        if batch.shape[0] > remaining:
            batch = batch[:remaining]
        accumulator.update(batch)

    return accumulator


def accumulate_generated_moments(model, features, num_samples, batch_size, stats_device):
    accumulator = MomentAccumulator(features=features, device=stats_device, dtype=torch.float64)

    with torch.inference_mode():
        while accumulator.n < num_samples:
            current_batch_size = min(batch_size, num_samples - accumulator.n)
            distribution = model()
            samples = distribution.sample((current_batch_size,))
            samples = samples.reshape(current_batch_size, -1)
            accumulator.update(samples)

    return accumulator


def _matrix_sqrt_psd(matrix, eps):
    matrix = (matrix + matrix.T) * 0.5
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    eigenvalues = eigenvalues.clamp_min(eps)
    return (eigenvectors * eigenvalues.sqrt().unsqueeze(0)) @ eigenvectors.T


def frechet_distance(mean_real, cov_real, mean_generated, cov_generated, sqrt_eps=0.0):
    diff = mean_real - mean_generated
    cov_real_sqrt = _matrix_sqrt_psd(cov_real, sqrt_eps)
    middle = cov_real_sqrt @ cov_generated @ cov_real_sqrt
    middle_sqrt = _matrix_sqrt_psd(middle, sqrt_eps)
    distance = diff.dot(diff) + torch.trace(cov_real + cov_generated - 2.0 * middle_sqrt)
    return distance.clamp_min(0.0)


def validate(
    base_model_name,
    base_layer_name,
    base_dataset_name,
    base_dataset_split,
    base_dataset_config,
    activations_base_path,
    gen_model_name,
    gen_model_type,
    checkpoint,
    device,
    batch_size,
    sample_batch_size,
    num_samples,
    stats_device,
    sqrt_eps,
    seed,
):
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    if batch_size <= 0:
        raise ValueError("--batch_size must be positive")
    if sample_batch_size is not None and sample_batch_size <= 0:
        raise ValueError("--sample_batch_size must be positive")

    dataset = ActivationDataset(
        model_name=base_model_name,
        layer=base_layer_name,
        dataset=base_dataset_name,
        split=base_dataset_split,
        dataset_config=base_dataset_config,
        base_dir=activations_base_path,
        device="cpu",
        dtype=torch.float32,
    )

    if num_samples is None:
        num_samples = len(dataset)
    if num_samples < 2:
        raise ValueError("--num_samples must be at least 2")
    if num_samples > len(dataset):
        raise ValueError(
            f"--num_samples={num_samples} exceeds real dataset length {len(dataset)}"
        )

    save_dir = make_path(
        make_activation_dataset_dir_path(
            activations_base_path,
            base_dataset_name,
            base_dataset_config,
            base_dataset_split,
        ),
        base_model_name,
        base_layer_name,
        "gen_models",
        gen_model_name,
    )
    checkpoint_path = resolve_checkpoint(save_dir, checkpoint)

    features = dataset[0].shape[0]
    model = load_generative_model(
        gen_model_name=gen_model_name,
        gen_model_type=gen_model_type,
        features=features,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    sample_batch_size = sample_batch_size or batch_size
    real_accumulator = accumulate_real_moments(
        dataset=dataset,
        num_samples=num_samples,
        batch_size=batch_size,
        stats_device=stats_device,
    )
    generated_accumulator = accumulate_generated_moments(
        model=model,
        features=features,
        num_samples=num_samples,
        batch_size=sample_batch_size,
        stats_device=stats_device,
    )

    mean_real, cov_real = real_accumulator.mean_and_cov()
    mean_generated, cov_generated = generated_accumulator.mean_and_cov()
    distance = frechet_distance(
        mean_real=mean_real,
        cov_real=cov_real,
        mean_generated=mean_generated,
        cov_generated=cov_generated,
        sqrt_eps=sqrt_eps,
    )

    result = {
        "frechet_distance": float(distance.cpu()),
        "num_real_samples": real_accumulator.n,
        "num_generated_samples": generated_accumulator.n,
        "features": features,
        "checkpoint": str(checkpoint_path),
        "hdf5_path": str(dataset.h5_path),
        "hdf5_dataset": dataset.hdf5_dataset,
    }
    print(json.dumps(result, indent=2))
    dataset.close()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name", type=str, required=True, help="Name of the model whose activations are being validated")
    parser.add_argument("--base_layer_name", type=str, required=True, help="Name of the layer whose activations are being validated")
    parser.add_argument("--base_dataset_name", type=str, required=True, help="Name of the dataset whose activations are being validated")
    parser.add_argument("--base_dataset_split", type=str, required=True, help="Dataset split for the activation HDF5 file")
    parser.add_argument("--base_dataset_config", type=str, default=None, help="Dataset config for the activation HDF5 file")
    parser.add_argument("--activations_base_path", type=str, default="/mnt/storage/yolcu/geocanon-activations", help="Path to the activation data folder")
    parser.add_argument("--gen_model_name", type=str, required=True, help="Name of the trained generative model descriptor")
    parser.add_argument("--gen_model_type", choices=["nf"], required=True, help="Type of generative model")
    parser.add_argument("--checkpoint", type=str, default="final", help="Checkpoint to load: final/latest, epoch number, filename, or path")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for the generative model")
    parser.add_argument("--batch_size", type=int, default=4096, help="Batch size for reading real activations")
    parser.add_argument("--sample_batch_size", type=int, default=None, help="Batch size for generated samples; defaults to --batch_size")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of real/generated samples to compare; defaults to the full HDF5 dataset")
    parser.add_argument("--stats_device", type=str, default="cpu", help="Device used for moment and covariance accumulation")
    parser.add_argument("--sqrt_eps", type=float, default=0.0, help="Minimum eigenvalue used in covariance square roots")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for generated samples; pass -1 to leave unchanged")
    args = parser.parse_args()

    validate(
        base_model_name=args.base_model_name,
        base_layer_name=args.base_layer_name,
        base_dataset_name=args.base_dataset_name,
        base_dataset_split=args.base_dataset_split,
        base_dataset_config=args.base_dataset_config,
        activations_base_path=args.activations_base_path,
        gen_model_name=args.gen_model_name,
        gen_model_type=args.gen_model_type,
        checkpoint=args.checkpoint,
        device=args.device,
        batch_size=args.batch_size,
        sample_batch_size=args.sample_batch_size,
        num_samples=args.num_samples,
        stats_device=args.stats_device,
        sqrt_eps=args.sqrt_eps,
        seed=None if args.seed < 0 else args.seed,
    )
