from pathlib import Path
from typing import Callable, Sequence

import torch
import torch.nn as nn
import zuko
import json

from torch import Tensor
from torch.distributions import Distribution, Independent, Normal
from torch.utils.data import DataLoader
from tqdm import tqdm

class GaussianEncoder(zuko.lazy.LazyDistribution):
    """
    q(z | x) = N(mu(x), diag(sigma(x)^2))
    """

    def __init__(
        self,
        features: int,
        context: int,
        hidden_features: Sequence[int] = (1024, 1024),
    ) -> None:
        super().__init__()

        layers = []
        input_features = context

        for hidden in hidden_features:
            layers.extend([
                nn.Linear(input_features, hidden),
                nn.ReLU(),
            ])
            input_features = hidden

        # Outputs latent mean and latent log standard deviation.
        layers.append(nn.Linear(input_features, 2 * features))

        self.hyper = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Distribution:
        parameters = self.hyper(x)
        mean, log_std = parameters.chunk(2, dim=-1)

        return Independent(
            Normal(mean, log_std.exp()),
            1,
        )

class FixedGaussianDecoder(zuko.lazy.LazyDistribution):

    """
    p(x | z) = N(mu(z), decoder_std^2 I)
    """

    def __init__(
        self,
        features: int,
        context: int,
        hidden_features: Sequence[int] = (1024, 1024),
    ) -> None:
        super().__init__()

        layers = []
        input_features = context

        for hidden in hidden_features:
            layers.extend([
                nn.Linear(input_features, hidden),
                nn.ReLU(),
            ])
            input_features = hidden

        layers.append(nn.Linear(input_features, features))

        self.hyper = nn.Sequential(*layers)

    def forward(self, z: Tensor) -> Distribution:
        mean = self.hyper(z)

        return Independent(
            Normal(mean, 1.0),
            1,
        )


def make_prior(
    latent_dim: int,
    name: str
) -> zuko.lazy.UnconditionalDistribution:
    return {"std":zuko.lazy.UnconditionalDistribution(
        zuko.distributions.DiagNormal,
        torch.zeros(latent_dim),
        torch.ones(latent_dim),
        buffer=True,
    )}[name]

class GaussianVAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        prior: str,
        encoder_hidden_features: Sequence[int] = (1024, 1024),
        decoder_hidden_features: Sequence[int] = (1024, 1024),
        beta: float = 1.0,
    ) -> None:
        super().__init__()

        self.encoder = GaussianEncoder(
                features=latent_dim,
                context=input_dim,
                hidden_features=encoder_hidden_features,
            )
        self.decoder = FixedGaussianDecoder(
            features=input_dim,
            context=latent_dim,
            hidden_features=decoder_hidden_features,
        )
        self.prior = make_prior(latent_dim, prior)
        self.register_buffer(
            "beta",
            torch.tensor(beta, dtype=torch.get_default_dtype()),
        )

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        q = self.encoder(x)
        z = q.rsample()

        p_x_given_z = self.decoder(z)

        # Shape: [batch]
        reconstruction_log_likelihood = p_x_given_z.log_prob(x)

        # q is Independent(Normal(...), 1).
        mean = q.base_dist.loc
        std = q.base_dist.scale
        log_std = std.log()

        if isinstance(self.prior, zuko.lazy.UnconditionalDistribution):
            # Exact KL(q(z | x) || N(0, I)).
            kl = 0.5 * (
                mean.square()
                + std.square()
                - 1.0 # becomes -1*latent_dim when summed over latent dimension
                - 2.0 * log_std # -2 because std instead of variance
            ).sum(dim=-1)
        else:
            # Monte Carlo estimate of KL(q(z | x) || p(z)).
            kl = q.log_prob(z) - self.prior.log_prob(z)

        negative_elbo = (
            -reconstruction_log_likelihood
            + self.beta * kl
        )

        reconstruction_mean = p_x_given_z.mean

        return {
            "loss": negative_elbo.mean(),
            "reconstruction_nll": (
                -reconstruction_log_likelihood
            ).mean(),
            "mse": (
                reconstruction_mean - x
            ).square().mean(),
            "kl": kl.mean(),
        }


def evaluate_loss(
    vae: GaussianVAE,
    ldr: DataLoader,
    device: torch.device | str,
) -> dict[str, float]:
    vae.eval()
    totals = {
        "loss": 0.0,
        "reconstruction_nll": 0.0,
        "mse": 0.0,
        "kl": 0.0,
    }
    number_of_examples = 0

    with torch.no_grad():
        for batch in ldr:
            x = batch.to(device=device)
            terms = vae(x)
            current_batch_size = x.shape[0]
            number_of_examples += current_batch_size

            for name in totals:
                totals[name] += (
                    terms[name].detach().item()
                    * current_batch_size
                )

    vae.train()

    return {
        name: value / number_of_examples
        for name, value in totals.items()
    }


def train_vae(
    train_data,
    optimizer_cls,
    lr: float,
    batch_size: int,
    epochs: int,
    device: torch.device | str,
    model_args: dict = {},
    optimizer_args: dict = {},
    beta: float = 1.0,
    save_freq: int = 0,
    save_dir: str | Path | None = None,
    test_data=None,
    checkpoint_callback: Callable[[str], None] | None = None,
):
    activation_dim = train_data[0].shape[0]

    vae = GaussianVAE(
        input_dim=activation_dim,
        beta=beta,
        **model_args
    ).to(device)

    optimizer = optimizer_cls(
        vae.parameters(),
        lr=lr,
        **optimizer_args,
    )

    loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
    )
    test_eval_loader = (
        DataLoader(
            test_data,
            batch_size=batch_size,
            shuffle=False,
        )
        if test_data is not None
        else None
    )

    save_path = Path(save_dir) if save_dir is not None else None
    vae.to(device)
    vae.train()

    history = []

    def save_checkpoint(epoch: int) -> tuple[str, dict[str, float] | None]:
        checkpoint_name = f"epoch{epoch}.pt"
        train_metrics = evaluate_loss(vae, loader, device)
        test_metrics = (
            evaluate_loss(vae, test_eval_loader, device)
            if test_eval_loader is not None
            else None
        )
        torch.save(
            vae.state_dict(),
            save_path / checkpoint_name,
        )
        return checkpoint_name, train_metrics, test_metrics

    for epoch in range(epochs):
        number_of_examples = 0

        progress = tqdm(
            loader,
            desc=f"Epoch {epoch + 1}/{epochs}",
        )

        for batch in progress:
            x = batch.to(device=device)

            optimizer.zero_grad(set_to_none=True)

            terms = vae(x)

            terms["loss"].backward()
            optimizer.step()

            current_batch_size = x.shape[0]
            number_of_examples += current_batch_size

            progress.set_postfix(
                total_loss=terms["loss"].item(),
                recon_nll=terms["reconstruction_nll"].item(),
                mse=terms["mse"].item(),
                kl=terms["kl"].item(),
            )

        epoch_number = epoch + 1
        should_save_checkpoint = (
            save_path is not None
            and (
                epoch_number == epochs
                or (save_freq > 0 and epoch_number % save_freq == 0)
            )
        )
        if should_save_checkpoint:
            checkpoint_name, train_metrics, test_metrics = save_checkpoint(epoch_number)
            if checkpoint_callback is not None:
                checkpoint_callback(checkpoint_name)
            history.append({
                "epoch": epoch_number,
                "checkpoint": checkpoint_name,
                "train_loss": train_metrics["loss"],
                "train_metrics": train_metrics,
                "test_loss": test_metrics["loss"],
                "test_metrics": test_metrics,
            })

    if save_path is not None:
        with (save_path / "checkpoint_losses.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    return vae, history
