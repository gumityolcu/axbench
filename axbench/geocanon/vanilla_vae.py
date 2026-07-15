import json
import os
import hashlib
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from axbench.geocanon.generative.vae import ClassifiedGaussianVAE
from axbench.utils.model_utils import gather_residual_activations


ARCHITECTURES = {
    "small": {
        "latent_dim": 64,
        "encoder_hidden_features": (512,),
        "decoder_hidden_features": (512,),
    },
    "big": {
        "latent_dim": 256,
        "encoder_hidden_features": (1024, 1024),
        "decoder_hidden_features": (1024, 1024),
    },
}


def _safe_path_part(value: Any) -> str:
    safe = str(value).replace(os.sep, "__").replace("/", "__")
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in safe)


def _atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def model_dir(dump_dir: str | Path, model_name: str = "VanillaVAE") -> Path:
    path = Path(dump_dir) / model_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def activation_cache_path(
    dump_dir: str | Path,
    layer: int,
    cache_kind: str,
    cache_id: str | int,
    model_name: str = "VanillaVAE",
) -> Path:
    return (
        model_dir(dump_dir, model_name)
        / "activations"
        / f"layer_{layer}"
        / cache_kind
        / f"{_safe_path_part(cache_id)}.pt"
    )


def concept_dir(
    dump_dir: str | Path,
    concept_id: int,
    model_name: str = "VanillaVAE",
) -> Path:
    path = model_dir(dump_dir, model_name) / "concepts" / f"concept_{concept_id:06d}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def checkpoint_path(
    dump_dir: str | Path,
    concept_id: int,
    model_name: str = "VanillaVAE",
) -> Path:
    return concept_dir(dump_dir, concept_id, model_name) / "model.pt"


def _fingerprint_texts(examples, extra: dict[str, Any]) -> str:
    digest = hashlib.sha256(json.dumps(extra, sort_keys=True).encode("utf-8"))
    for text in examples["input"].astype(str):
        encoded = text.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()[:16]


def _metadata_for_concept(metadata_path: str | Path, concept_id: int) -> dict[str, Any]:
    with Path(metadata_path).open("r", encoding="utf-8") as metadata_file:
        for line in metadata_file:
            record = json.loads(line)
            if int(record["concept_id"]) == int(concept_id):
                return record
    raise ValueError(f"Concept {concept_id} was not found in {metadata_path}")


def _concept_genre(metadata: dict[str, Any]) -> str:
    concept = metadata["concept"]
    genres = metadata["concept_genres_map"][concept]
    if not genres:
        raise ValueError(f"Concept {concept!r} has no genre in metadata")
    return genres[0]


def _extract_token_activations(
    model,
    tokenizer,
    examples,
    layer: int,
    prefix_length: int,
    batch_size: int,
    device: torch.device | str,
) -> torch.Tensor:
    if len(examples) == 0:
        hidden_size = model.config.hidden_size
        return torch.empty((0, hidden_size), dtype=torch.float32)

    rows = []
    for start in tqdm(range(0, len(examples), batch_size), desc="Extracting VAE activations"):
        batch = examples.iloc[start:start + batch_size]
        inputs = tokenizer(
            batch["input"].tolist(),
            return_tensors="pt",
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(device)
        with torch.inference_mode():
            activations = gather_residual_activations(model, layer, inputs).detach()
        attention_mask = inputs["attention_mask"][:, prefix_length:].bool()
        batch_rows = activations[:, prefix_length:][attention_mask]
        rows.append(batch_rows.float().cpu())
        del activations
        torch.cuda.empty_cache()

    if not rows:
        hidden_size = model.config.hidden_size
        return torch.empty((0, hidden_size), dtype=torch.float32)
    return torch.cat(rows, dim=0)


def load_or_extract_activations(
    model,
    tokenizer,
    examples,
    layer: int,
    prefix_length: int,
    batch_size: int,
    device: torch.device | str,
    cache_path: Path,
    manifest: dict[str, Any],
) -> torch.Tensor:
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu")
        if payload.get("manifest", {}).get("fingerprint") == manifest["fingerprint"]:
            return payload["activations"].float()

    activations = _extract_token_activations(
        model=model,
        tokenizer=tokenizer,
        examples=examples,
        layer=layer,
        prefix_length=prefix_length,
        batch_size=batch_size,
        device=device,
    )
    payload = {
        "activations": activations,
        "manifest": {
            **manifest,
            "n_rows": int(activations.shape[0]),
            "activation_dim": int(activations.shape[1]) if activations.ndim == 2 else None,
        },
    }
    _atomic_torch_save(payload, cache_path)
    return activations


def _architecture_config(params: Any) -> dict[str, Any]:
    architecture = getattr(params, "vae_architecture", None) or "small"
    if architecture not in ARCHITECTURES:
        raise ValueError(
            f"Unsupported VanillaVAE architecture {architecture!r}; "
            f"choose one of {sorted(ARCHITECTURES)}"
        )
    config = dict(ARCHITECTURES[architecture])
    latent_dim = getattr(params, "vae_latent_dim", None)
    if latent_dim is not None:
        config["latent_dim"] = int(latent_dim)
    config["architecture"] = architecture
    return config


def _parameter(params: Any, name: str, default: Any) -> Any:
    value = getattr(params, name, None)
    return default if value is None else value


def make_vanilla_vae(input_dim: int, params: Any) -> ClassifiedGaussianVAE:
    arch = _architecture_config(params)
    return ClassifiedGaussianVAE(
        input_dim=input_dim,
        latent_dim=arch["latent_dim"],
        prior="std",
        encoder_hidden_features=arch["encoder_hidden_features"],
        decoder_hidden_features=arch["decoder_hidden_features"],
        beta=float(_parameter(params, "vae_kl_lambda", 1.0)),
        reconstruction_lambda=float(_parameter(params, "vae_reconstruction_lambda", 1.0)),
        classification_lambda=float(_parameter(params, "vae_classification_lambda", 1.0)),
    )


def load_auxiliary_texts(
    source: str | Path | pd.DataFrame,
    text_column: str = "text",
    max_examples: int | None = None,
) -> pd.DataFrame:
    """Load a local text corpus and normalize it to the model's `input` column."""
    if isinstance(source, pd.DataFrame):
        data = source.copy()
    else:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Auxiliary VAE data was not found: {path}")
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            data = pd.read_parquet(path)
        elif suffix in {".csv", ".tsv"}:
            data = pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",")
        elif suffix in {".jsonl", ".ndjson"}:
            data = pd.read_json(path, lines=True)
        elif suffix == ".json":
            data = pd.read_json(path)
        elif suffix in {".txt", ".text"}:
            with path.open("r", encoding="utf-8") as text_file:
                data = pd.DataFrame({text_column: [line.strip() for line in text_file if line.strip()]})
        else:
            raise ValueError(
                f"Unsupported auxiliary data format {suffix!r}; use parquet, csv/tsv, "
                "json/jsonl, or one-text-per-line txt"
            )

    if text_column not in data.columns and text_column == "text" and "input" in data.columns:
        text_column = "input"
    if text_column not in data.columns:
        raise ValueError(
            f"Auxiliary data has no {text_column!r} column; available columns: {list(data.columns)}"
        )
    texts = data[text_column].dropna().astype(str)
    texts = texts[texts.str.strip().astype(bool)]
    if max_examples is not None:
        texts = texts.iloc[:int(max_examples)]
    if texts.empty:
        raise ValueError("Auxiliary VAE data contains no non-empty texts")
    return pd.DataFrame({"input": texts.reset_index(drop=True)})


def train_vanilla_vae(
    model,
    tokenizer,
    examples,
    metadata_path: str | Path,
    concept_id: int,
    layer: int,
    dump_dir: str | Path,
    training_args: Any,
    prefix_length: int,
    device: torch.device | str,
    lm_model_name: str | None = None,
    auxiliary_examples=None,
    auxiliary_prefix_length: int = 1,
    model_name: str = "VanillaVAE",
) -> dict[str, Any]:
    metadata = _metadata_for_concept(metadata_path, concept_id)
    genre = _concept_genre(metadata)

    missing_columns = {"input", "labels"} - set(examples.columns)
    if missing_columns:
        raise ValueError(
            f"{model_name} requires labeled text columns 'input' and 'labels'; "
            f"missing {sorted(missing_columns)}"
        )
    positive_df = examples[examples["labels"] == 1].copy()
    negative_df = examples[examples["labels"] == 0].copy()
    activation_batch_size = int(
        getattr(training_args, "vae_activation_batch_size", None)
        or getattr(training_args, "batch_size", None)
        or 8
    )

    cache_context = {
        "layer": int(layer),
        "model_name": lm_model_name,
        "tokenizer": getattr(tokenizer, "name_or_path", tokenizer.__class__.__name__),
        "prefix_length": int(prefix_length),
    }
    positive_manifest = {
        **cache_context,
        "kind": "positive",
        "concept_id": int(concept_id),
        "concept": metadata["concept"],
        "genre": genre,
    }
    positive_manifest["fingerprint"] = _fingerprint_texts(positive_df, positive_manifest)
    negative_manifest = {
        **cache_context,
        "kind": "negative",
        "genre": genre,
    }
    negative_manifest["fingerprint"] = _fingerprint_texts(negative_df, negative_manifest)
    positive_cache = activation_cache_path(
        dump_dir, layer, "positives", f"{concept_id}_{positive_manifest['fingerprint']}", model_name
    )
    negative_cache = activation_cache_path(
        dump_dir, layer, "negatives_by_genre", f"{genre}_{negative_manifest['fingerprint']}", model_name
    )
    positive_activations = load_or_extract_activations(
        model=model,
        tokenizer=tokenizer,
        examples=positive_df,
        layer=layer,
        prefix_length=prefix_length,
        batch_size=activation_batch_size,
        device=device,
        cache_path=positive_cache,
        manifest=positive_manifest,
    )
    negative_activations = load_or_extract_activations(
        model=model,
        tokenizer=tokenizer,
        examples=negative_df,
        layer=layer,
        prefix_length=prefix_length,
        batch_size=activation_batch_size,
        device=device,
        cache_path=negative_cache,
        manifest=negative_manifest,
    )

    if positive_activations.numel() == 0 or negative_activations.numel() == 0:
        raise ValueError(
            f"{model_name} needs positive and negative activations for concept {concept_id}; "
            f"got {positive_activations.shape[0]} positive and {negative_activations.shape[0]} negative rows"
        )

    x = torch.cat([positive_activations, negative_activations], dim=0).float()
    y = torch.cat([
        torch.ones(positive_activations.shape[0]),
        torch.zeros(negative_activations.shape[0]),
    ], dim=0)
    generator = torch.Generator()
    generator.manual_seed(int(_parameter(training_args, "seed", 42)))
    batch_size = int(_parameter(training_args, "batch_size", 64))
    labeled_loader = DataLoader(
        TensorDataset(x, y),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )

    reconstruction_loader = None
    auxiliary_cache = None
    if auxiliary_examples is not None:
        auxiliary_manifest = {
            **cache_context,
            "kind": "auxiliary",
            "prefix_length": int(auxiliary_prefix_length),
        }
        auxiliary_manifest["fingerprint"] = _fingerprint_texts(
            auxiliary_examples, auxiliary_manifest
        )
        auxiliary_cache = activation_cache_path(
            dump_dir,
            layer,
            "auxiliary",
            auxiliary_manifest["fingerprint"],
            model_name,
        )
        auxiliary_activations = load_or_extract_activations(
            model=model,
            tokenizer=tokenizer,
            examples=auxiliary_examples,
            layer=layer,
            prefix_length=auxiliary_prefix_length,
            batch_size=activation_batch_size,
            device=device,
            cache_path=auxiliary_cache,
            manifest=auxiliary_manifest,
        )
        if auxiliary_activations.numel() == 0:
            raise ValueError("Auxiliary VAE data produced no token activations")
        reconstruction_loader = DataLoader(
            TensorDataset(auxiliary_activations),
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
        )

    vae = make_vanilla_vae(input_dim=x.shape[1], params=training_args).to(device)
    optimizer = torch.optim.AdamW(
        vae.parameters(),
        lr=float(_parameter(training_args, "lr", 1e-3)),
        weight_decay=float(_parameter(training_args, "weight_decay", 0.0)),
    )
    epochs = int(_parameter(training_args, "n_epochs", 20))
    history = []
    vae.train()
    for epoch in range(epochs):
        totals = {
            "loss": 0.0,
            "reconstruction_nll": 0.0,
            "mse": 0.0,
            "kl": 0.0,
            "classification_loss": 0.0,
            "classification_accuracy": 0.0,
        }
        n_seen = 0
        steps = len(reconstruction_loader) if reconstruction_loader is not None else len(labeled_loader)
        labeled_iterator = iter(labeled_loader)
        reconstruction_iterator = iter(reconstruction_loader) if reconstruction_loader is not None else None
        progress = tqdm(range(steps), desc=f"{model_name} concept {concept_id} epoch {epoch + 1}/{epochs}")
        for _ in progress:
            try:
                batch_x, batch_y = next(labeled_iterator)
            except StopIteration:
                labeled_iterator = iter(labeled_loader)
                batch_x, batch_y = next(labeled_iterator)
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            if reconstruction_iterator is None:
                terms = vae(batch_x, batch_y)
            else:
                (reconstruction_x,) = next(reconstruction_iterator)
                reconstruction_x = reconstruction_x.to(device)
                reconstruction_terms = vae.reconstruction_terms(reconstruction_x)
                classification_terms = vae.classification_terms(batch_x, batch_y)
                loss = vae.objective(
                    reconstruction_terms,
                    classification_terms["classification_loss"],
                )
                terms = {"loss": loss, **reconstruction_terms, **classification_terms}
            optimizer.zero_grad(set_to_none=True)
            terms["loss"].backward()
            optimizer.step()

            batch_n = batch_x.shape[0]
            n_seen += batch_n
            for key in totals:
                totals[key] += float(terms[key].detach().cpu()) * batch_n
            progress.set_description(
                f"{model_name} concept {concept_id} loss {float(terms['loss'].detach().cpu()):.4f}"
            )
        history.append({key: value / n_seen for key, value in totals.items()})

    ckpt = {
        "state_dict": vae.state_dict(),
        "concept_id": int(concept_id),
        "concept": metadata["concept"],
        "genre": genre,
        "input_dim": int(x.shape[1]),
        "architecture": _architecture_config(training_args),
        "params": {
            "batch_size": batch_size,
            "n_epochs": epochs,
            "lr": float(_parameter(training_args, "lr", 1e-3)),
            "weight_decay": float(_parameter(training_args, "weight_decay", 0.0)),
            "vae_kl_lambda": float(_parameter(training_args, "vae_kl_lambda", 1.0)),
            "vae_reconstruction_lambda": float(_parameter(training_args, "vae_reconstruction_lambda", 1.0)),
            "vae_classification_lambda": float(_parameter(training_args, "vae_classification_lambda", 1.0)),
        },
        "activation_caches": {
            "positive": str(positive_cache),
            "negative": str(negative_cache),
            "auxiliary": str(auxiliary_cache) if auxiliary_cache is not None else None,
        },
        "history": history,
    }
    out_dir = concept_dir(dump_dir, concept_id, model_name)
    _atomic_torch_save(ckpt, out_dir / "model.pt")
    with (out_dir / "config.json").open("w", encoding="utf-8") as config_file:
        json.dump({k: v for k, v in ckpt.items() if k != "state_dict"}, config_file, indent=2)

    classifier = vae.classifier
    return {
        "checkpoint_path": str(out_dir / "model.pt"),
        "history": history,
        "classifier_weight": classifier.weight.detach().cpu(),
        "classifier_bias": classifier.bias.detach().cpu(),
    }


def load_vanilla_vae(
    dump_dir: str | Path,
    concept_id: int,
    device: torch.device | str,
    model_name: str = "VanillaVAE",
):
    ckpt = torch.load(checkpoint_path(dump_dir, concept_id, model_name), map_location="cpu")
    params = type("VanillaVAEParams", (), {})()
    setattr(params, "vae_architecture", ckpt["architecture"]["architecture"])
    setattr(params, "vae_latent_dim", ckpt["architecture"]["latent_dim"])
    for key, value in ckpt["params"].items():
        setattr(params, key, value)
    vae = make_vanilla_vae(input_dim=ckpt["input_dim"], params=params)
    vae.load_state_dict(ckpt["state_dict"])
    vae.to(device)
    vae.eval()
    return vae, ckpt
