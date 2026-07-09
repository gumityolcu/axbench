import argparse
import json
import os
from pathlib import Path

import h5py
import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader

from utils.HF_handlers.HF_dataset_handler import load_dataset
from utils.HF_handlers.HF_model_handler import HFModelAdapter
from utils.misc import make_path

load_dotenv()


def _to_hdf5_rows(hidden, token_mask):
    rows = hidden[token_mask].contiguous()
    if rows.dtype == torch.bfloat16:
        rows = rows.float()
    return rows.numpy()


def _append_rows(h5_dataset, rows):
    start = h5_dataset.shape[0]
    end = start + rows.shape[0]
    h5_dataset.resize((end, rows.shape[1]))
    h5_dataset[start:end] = rows
    return start, end


def main(
    model_name,
    layers,
    dataset,
    dataset_config,
    split,
    device,
    save_dir,
    stream_dataset=False,
    batch_size=8,
    subset=None
):
    if not layers:
        raise ValueError("At least one layer must be provided with --layers")

    token = os.environ.get("HF_TOKEN")
    if token is None:
        print("HF_TOKEN is not set; continuing with public model/dataset access only.")

    model_adapter = HFModelAdapter(model_name, device_map={"": device})
    model = model_adapter.model
    model.eval()

    acts = {}

    def save_layer_output(layer_name):
        def hook(module, inputs, output):
            # Many HF decoder blocks return either a tensor or a tuple whose first
            # item is the hidden state tensor.
            hidden = output[0] if isinstance(output, tuple) else output
            acts[layer_name] = hidden.detach().cpu()

        return hook

    modules = dict(model.named_modules())
    missing_layers = [name for name in layers if name not in modules]
    if missing_layers:
        available = ", ".join(list(modules.keys())[:20])
        raise KeyError(
            f"Module(s) not found: {missing_layers}. First available modules: {available}"
        )

    ds, collate_fn = load_dataset(
        name=dataset,
        tokenizer=model_adapter.tokenizer,
        config=dataset_config,
        split=split,
        streaming=stream_dataset,
    )

    if subset is not None:
        if subset < 0:
            raise ValueError("--subset must be non-negative")
        ds = ds.take(subset)

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    layer_outputs = {}
    dataset_dir = make_path(save_dir, dataset, dataset_config or "default", split)
    token_manifest_path = dataset_dir / "token_manifest.jsonl"
    tokens_per_example = []
    for layer_name in layers:
        out_dir = make_path(dataset_dir, model_name, layer_name)
        h5_path = out_dir / "activations.hdf5"
        manifest_path = out_dir / "manifest.jsonl"
        h5_file = h5py.File(h5_path, "w")
        layer_outputs[layer_name] = {
            "dir": out_dir,
            "h5_path": h5_path,
            "manifest_path": manifest_path,
            "h5_file": h5_file,
            "dataset": None,
            "dtype": None,
        }

    handles = [
        modules[layer_name].register_forward_hook(save_layer_output(layer_name))
        for layer_name in layers
    ]

    try:
        with torch.inference_mode():
            for batch_idx, batch in enumerate(loader):
                acts.clear()
                batch = {k: v.to(model.device) for k, v in batch.items()}
                attention_mask = batch.get("attention_mask")
                if attention_mask is None:
                    raise KeyError("Batch is missing attention_mask; cannot filter padding tokens")

                _ = model(**batch, use_cache=False)
                token_mask = attention_mask.detach().cpu().bool()
                batch_tokens_per_example = token_mask.sum(dim=1).tolist()
                tokens_per_example.extend(batch_tokens_per_example)
                for layer_name in layers:
                    if layer_name not in acts:
                        raise RuntimeError(f"Hook for layer {layer_name} did not capture activations")

                    hidden = acts[layer_name]
                    if hidden.ndim != 3:
                        raise ValueError(
                            f"Expected layer {layer_name} output to have shape "
                            f"[batch, seq, hidden], got {tuple(hidden.shape)}"
                        )
                    if hidden.shape[:2] != token_mask.shape:
                        raise ValueError(
                            f"Activation/token mask shape mismatch for {layer_name}: "
                            f"{tuple(hidden.shape[:2])} vs {tuple(token_mask.shape)}"
                        )

                    rows = _to_hdf5_rows(hidden, token_mask)
                    output = layer_outputs[layer_name]
                    if output["dataset"] is None:
                        output["dataset"] = output["h5_file"].create_dataset(
                            "activations",
                            shape=(0, rows.shape[1]),
                            maxshape=(None, rows.shape[1]),
                            dtype=rows.dtype,
                            chunks=(max(1, min(8192, rows.shape[0])), rows.shape[1]),
                        )
                        output["dtype"] = str(rows.dtype)

                    _append_rows(output["dataset"], rows)

                if (batch_idx + 1) % 10 == 0:
                    total = sum(tokens_per_example)
                    print(f"Processed {batch_idx + 1} batches / {total} tokens")
    finally:
        for handle in handles:
            handle.remove()
        for output in layer_outputs.values():
            output["h5_file"].close()

    with token_manifest_path.open("w", encoding="utf-8") as manifest_file:
        start_row = 0
        for example_idx, n_tokens in enumerate(tokens_per_example):
            end_row = start_row + n_tokens
            record = {
                "example_idx": example_idx,
                "n_tokens": n_tokens,
                "start_row": start_row,
                "end_row": end_row,
            }
            manifest_file.write(json.dumps(record) + "\n")
            start_row = end_row

    for layer_name, output in layer_outputs.items():
        record = {
            "path": str(output["h5_path"]),
            "hdf5_dataset": "activations",
            "dtype": output["dtype"],
            "n_rows": sum(tokens_per_example),
            "layer": layer_name,
        }
        with output["manifest_path"].open("w", encoding="utf-8") as manifest_file:
            manifest_file.write(json.dumps(record) + "\n")
        print(f"Saved {layer_name}: {output['h5_path']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--layers", type=str, nargs="+", required=True)
    parser.add_argument("--dataset", type=str, default="roneneldan/TinyStories")
    parser.add_argument("--dataset_config", type=str, default=None)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=str, default="/mnt/storage/yolcu/geocanon-activations")
    parser.add_argument("--stream_dataset", action="store_true")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--subset", type=int, default=None)
    args = parser.parse_args()
    main(
        model_name=args.model_name,
        layers=args.layers,
        dataset=args.dataset,
        dataset_config=args.dataset_config,
        split=args.split,
        device=args.device,
        save_dir=args.save_dir,
        stream_dataset=args.stream_dataset,
        batch_size=args.batch_size,
        subset=args.subset, # DEBUG
    )
