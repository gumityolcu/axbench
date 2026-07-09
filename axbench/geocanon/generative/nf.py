import json
from pathlib import Path
from typing import Callable

import torch
import zuko
from tqdm import tqdm

def load_flow_model_by_name(name):
    return {
    "nice": zuko.flows.NICE,
    }[name.lower()]

def evaluate_likelihood(flow, ldr, device, worst_k=10):
    was_training = flow.training
    flow.eval()

    all_log_probs = []
    all_base_log_probs = []
    all_log_dets = []

    with torch.no_grad():
        for batch in ldr:
            batch = batch.to(device)

            distribution = flow()
            transform = distribution.transform

            # Compute z = f(x) and log |det(df/dx)|.
            # call_and_ladj avoids evaluating the transformation twice.
            if hasattr(transform, "call_and_ladj"):
                z, log_det = transform.call_and_ladj(batch)
            else:
                z = transform(batch)
                log_det = transform.log_abs_det_jacobian(batch, z)

            base_log_prob = distribution.base.log_prob(z)

            if base_log_prob.shape != log_det.shape:
                raise RuntimeError(
                    "Unexpected component shapes: "
                    f"base_log_prob={base_log_prob.shape}, "
                    f"log_det={log_det.shape}"
                )

            log_prob = base_log_prob + log_det

            all_log_probs.append(log_prob.cpu())
            all_base_log_probs.append(base_log_prob.cpu())
            all_log_dets.append(log_det.cpu())

    flow.train(was_training)

    log_probs = torch.cat(all_log_probs).double()
    base_log_probs = torch.cat(all_base_log_probs).double()
    log_dets = torch.cat(all_log_dets).double()

    quantile_levels = torch.tensor(
        [
            0.0,
            0.0001,  # 0.01st percentile
            0.001,   # 0.1st percentile
            0.01,    # 1st percentile
            0.05,    # 5th percentile
            0.50,    # median
            0.95,
            0.99,
            1.0,
        ],
        dtype=torch.float64,
    )

    quantile_names = [
        "min",
        "p0.01",
        "p0.1",
        "p1",
        "p5",
        "median",
        "p95",
        "p99",
        "max",
    ]

    def summarize(values):
        finite_mask = torch.isfinite(values)
        finite_values = values[finite_mask]

        if finite_values.numel() == 0:
            return {
                "mean": None,
                "std": None,
                "nonfinite_count": values.numel(),
            }

        quantiles = torch.quantile(finite_values, quantile_levels)

        return {
            "mean": finite_values.mean().item(),
            "std": finite_values.std(unbiased=False).item(),
            "nonfinite_count": (~finite_mask).sum().item(),
            **{
                name: value.item()
                for name, value in zip(quantile_names, quantiles)
            },
        }

    k = min(worst_k, log_probs.numel())
    worst_indices = torch.topk(
        log_probs,
        k=k,
        largest=False,
    ).indices

    worst_examples = [
        {
            # This is the position in evaluation order. It equals the dataset
            # index when the DataLoader uses shuffle=False.
            "evaluation_index": index.item(),
            "log_prob": log_probs[index].item(),
            "base_log_prob": base_log_probs[index].item(),
            "log_det": log_dets[index].item(),
        }
        for index in worst_indices
    ]

    return {
        "count": log_probs.numel(),
        "log_prob": summarize(log_probs),
        "base_log_prob": summarize(base_log_probs),
        "log_det": summarize(log_dets),
        "worst_examples": worst_examples,
    }

def train_flow(
    train_data,
    test_data,
    model,
    optimizer_cls,
    lr,
    batch_size,
    epochs,
    device,
    optimizer_args={},
    model_args={},
    save_freq=0,
    save_dir=None,
    checkpoint_callback: Callable[[str], None] | None = None,
):
    dimension=train_data[0].shape[0]
    if isinstance(model, str):
        flow_cls = load_flow_model_by_name(model)
        flow = flow_cls(features=dimension, **model_args)
    else:
        flow = model

    optimizer = optimizer_cls(flow.parameters(), lr=lr, **optimizer_args)

    train_ldr = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    train_eval_ldr = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False)
    test_eval_ldr = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
    save_path = Path(save_dir) if save_dir is not None else None
    history = []
    flow.train()
    flow.to(device)

    def save_checkpoint(epoch):
        checkpoint_name = f"epoch{epoch}.pt"
        torch.save(flow.state_dict(), save_path / checkpoint_name)
        train_likelihood = evaluate_likelihood(flow, train_eval_ldr, device)
        test_likelihood = evaluate_likelihood(flow, test_eval_ldr, device)
        return checkpoint_name, train_likelihood, test_likelihood

    for e in range(epochs):
        total_log_prob = 0.0
        total_count = 0

        progress = tqdm(
            train_ldr,
            desc=f"Epoch {e + 1}/{epochs}",
        )
        for batch in progress:
            batch = batch.to(device)
            optimizer.zero_grad()
            log_prob = flow().log_prob(batch)
            loss = -log_prob.mean()
            loss.backward()
            optimizer.step()

            total_log_prob += log_prob.detach().sum().item()
            total_count += batch.shape[0]

            progress.set_postfix(
                total_loss=loss.detach().item(),
                nll=loss.detach().item(),
                log_prob=log_prob.detach().mean().item(),
            )

        epoch = e + 1
        should_save_checkpoint = (
            save_path is not None
            and (
                epoch == epochs
                or (save_freq > 0 and epoch % save_freq == 0)
            )
        )

        if should_save_checkpoint:
            checkpoint_name, train_likelihood, test_likelihood = save_checkpoint(epoch)
            if checkpoint_callback is not None:
                checkpoint_callback(checkpoint_name)

            history.append({
                "epoch": epoch,
                "checkpoint": checkpoint_name,
                "train_likelihood": train_likelihood,
                "test_likelihood": test_likelihood,
            })
    
    if save_path is not None:
        with (save_path / "checkpoint_likelihoods.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
