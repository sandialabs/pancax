from dataclasses import dataclass
from typing import Sequence
import torch

from .complex import complex_to
from .solver import (
    solve_dirichlet_nonlinear,
    adjoint_parameter_grads,
    assign_parameter_grads,
)


@dataclass
class Sample:
    meec: object
    rhs: torch.Tensor
    boundary_values: torch.Tensor
    target_u: torch.Tensor
    mu_scalar: torch.Tensor | None = None
    mu_vector: torch.Tensor | None = None
    forcing_features: torch.Tensor | None = None


def maybe_to(x, device):
    if x is None:
        return None
    return x.to(device)


def sample_to(sample: Sample, device):
    return Sample(
        meec=complex_to(sample.meec, device),
        rhs=sample.rhs.to(device),
        boundary_values=sample.boundary_values.to(device),
        target_u=sample.target_u.to(device),
        mu_scalar=maybe_to(sample.mu_scalar, device),
        mu_vector=maybe_to(sample.mu_vector, device),
        forcing_features=maybe_to(sample.forcing_features, device),
    )


def predict(model, sample: Sample, verbose: bool = False) -> torch.Tensor:
    device = next(model.parameters()).device
    sample = sample_to(sample, device)

    return solve_dirichlet_nonlinear(
        model=model,
        meec=sample.meec,
        rhs=sample.rhs,
        boundary_values=sample.boundary_values,
        mu_scalar=sample.mu_scalar,
        mu_vector=sample.mu_vector,
        forcing_features=sample.forcing_features,
        verbose=verbose,
    )


def train(
    model,
    dataset: Sequence[Sample],
    num_steps: int = 1000,
    lr: float = 5e-4,
    grad_clip: float = 1.0,
    print_every: int = 100,
    device: str = "cpu",
):
    device = torch.device(device)
    model.to(device)

    dataset = [sample_to(sample, device) for sample in dataset]

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_state = None
    best_loss = float("inf")

    for step in range(num_steps):
        optimizer.zero_grad()
        total_loss = 0.0
        num_samples = len(dataset)

        for i, sample in enumerate(dataset):
            u_star = solve_dirichlet_nonlinear(
                model=model,
                meec=sample.meec,
                rhs=sample.rhs,
                boundary_values=sample.boundary_values,
                mu_scalar=sample.mu_scalar,
                mu_vector=sample.mu_vector,
                forcing_features=sample.forcing_features,
                verbose=(step == 0 and i == 0),
            )

            loss, grads = adjoint_parameter_grads(
                model=model,
                u_star=u_star,
                target_u=sample.target_u,
                meec=sample.meec,
                rhs=sample.rhs,
                boundary_values=sample.boundary_values,
                mu_scalar=sample.mu_scalar,
                mu_vector=sample.mu_vector,
                forcing_features=sample.forcing_features,
                loss_reduction="mean",
            )

            total_loss += float(loss.item()) / num_samples
            grads = [g / num_samples for g in grads]
            assign_parameter_grads(model, grads, accumulate=True)

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        if total_loss < best_loss:
            best_loss = total_loss
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

        if step % print_every == 0:
            print(f"step={step}, loss={total_loss:.6e}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model
