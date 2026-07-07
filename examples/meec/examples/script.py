import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from meec.graph import build_epsilon_ball_graph
from meec.complex import build_meec_complex
from meec.model import FluxMLP, MEECNet
from meec.solver import solve_dirichlet_nonlinear
from meec.train import Sample, train


# ============================================================
# Utilities
# ============================================================

def make_square_points(nx=21, ny=21, dtype=torch.float64):
    xs = torch.linspace(0.0, 1.0, nx, dtype=dtype)
    ys = torch.linspace(0.0, 1.0, ny, dtype=dtype)
    X, Y = torch.meshgrid(xs, ys, indexing="xy")
    pts = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=-1)
    return pts


def make_boundary_mask(x: torch.Tensor, tol: float = 1e-12):
    return (
        (torch.abs(x[:, 0] - 0.0) < tol)
        | (torch.abs(x[:, 0] - 1.0) < tol)
        | (torch.abs(x[:, 1] - 0.0) < tol)
        | (torch.abs(x[:, 1] - 1.0) < tol)
    )


def save_solution_plot(
    x: torch.Tensor,
    u_pred: torch.Tensor,
    u_true: torch.Tensor | None = None,
    filename: str = "predicted_solution.pdf",
    title: str = "Predicted solution",
):
    x_np = x.detach().cpu().numpy()
    u_pred_np = u_pred.detach().cpu().numpy()
    triang = mtri.Triangulation(x_np[:, 0], x_np[:, 1])

    if u_true is None:
        fig, ax = plt.subplots(figsize=(6, 5))
        tpc = ax.tricontourf(triang, u_pred_np, levels=30, cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        fig.colorbar(tpc, ax=ax, label="u")
    else:
        u_true_np = u_true.detach().cpu().numpy()
        err_np = u_pred_np - u_true_np

        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)

        t0 = axes[0].tricontourf(triang, u_true_np, levels=30, cmap="viridis")
        axes[0].set_title("True solution")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        axes[0].set_aspect("equal")
        fig.colorbar(t0, ax=axes[0], label="u")

        t1 = axes[1].tricontourf(triang, u_pred_np, levels=30, cmap="viridis")
        axes[1].set_title("Predicted solution")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")
        axes[1].set_aspect("equal")
        fig.colorbar(t1, ax=axes[1], label="u")

        t2 = axes[2].tricontourf(triang, err_np, levels=30, cmap="coolwarm")
        axes[2].set_title("Prediction error")
        axes[2].set_xlabel("x")
        axes[2].set_ylabel("y")
        axes[2].set_aspect("equal")
        fig.colorbar(t2, ax=axes[2], label="u_pred - u_true")

        fig.suptitle(title)

    outpath = Path(filename)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Zero-flux model for Poisson validation
# ============================================================

class ZeroFluxNet(nn.Module):
    def forward(self, edge_features: torch.Tensor) -> torch.Tensor:
        return torch.zeros(edge_features.shape[0], dtype=edge_features.dtype, device=edge_features.device)


# ============================================================
# Manufactured solutions
# ============================================================

def manufactured_u(x: torch.Tensor, freq_x: int = 1, freq_y: int = 1) -> torch.Tensor:
    """
    Smooth solution with homogeneous Dirichlet boundary values on [0,1]^2:
        u(x,y) = sin(freq_x*pi*x) sin(freq_y*pi*y)
    """
    return (
        torch.sin(freq_x * math.pi * x[:, 0])
        * torch.sin(freq_y * math.pi * x[:, 1])
    )


def poisson_rhs(x: torch.Tensor, freq_x: int = 1, freq_y: int = 1) -> torch.Tensor:
    """
    For u = sin(a*pi*x) sin(b*pi*y):
        -Delta u = pi^2 (a^2 + b^2) sin(a*pi*x) sin(b*pi*y)
    """
    u = manufactured_u(x, freq_x=freq_x, freq_y=freq_y)
    return (math.pi ** 2) * (freq_x ** 2 + freq_y ** 2) * u


def advection_diffusion_rhs(
    x: torch.Tensor,
    kappa: float,
    velocity: torch.Tensor,
    freq_x: int = 1,
    freq_y: int = 1,
) -> torch.Tensor:
    """
    Manufactured RHS for:
        -kappa Delta u + v · grad u = f
    with
        u = sin(freq_x*pi*x) sin(freq_y*pi*y)
    """
    xx = x[:, 0]
    yy = x[:, 1]

    u = torch.sin(freq_x * math.pi * xx) * torch.sin(freq_y * math.pi * yy)
    ux = freq_x * math.pi * torch.cos(freq_x * math.pi * xx) * torch.sin(freq_y * math.pi * yy)
    uy = freq_y * math.pi * torch.sin(freq_x * math.pi * xx) * torch.cos(freq_y * math.pi * yy)

    diffusion_part = kappa * (math.pi ** 2) * (freq_x ** 2 + freq_y ** 2) * u
    advection_part = velocity[0] * ux + velocity[1] * uy
    return diffusion_part + advection_part


# ============================================================
# Problem factory
# ============================================================

@dataclass
class ManufacturedProblem:
    name: str
    rhs: torch.Tensor
    boundary_values: torch.Tensor
    target_u: torch.Tensor
    mu_scalar: torch.Tensor | None
    mu_vector: torch.Tensor | None
    flux_in_dim: int
    background_diffusion: float


def build_problem(
    x: torch.Tensor,
    boundary_mask: torch.Tensor,
    kind: str = "poisson",
    kappa: float = 0.1,
    velocity=(1.0, 0.5),
    freq_x: int = 1,
    freq_y: int = 1,
) -> ManufacturedProblem:
    target_u = manufactured_u(x, freq_x=freq_x, freq_y=freq_y)
    boundary_values = torch.where(boundary_mask, target_u, torch.zeros_like(target_u))

    if kind == "poisson":
        rhs = poisson_rhs(x, freq_x=freq_x, freq_y=freq_y)
        mu_scalar = None
        mu_vector = None
        flux_in_dim = 2
        background_diffusion = 1.0

        return ManufacturedProblem(
            name="poisson",
            rhs=rhs,
            boundary_values=boundary_values,
            target_u=target_u,
            mu_scalar=mu_scalar,
            mu_vector=mu_vector,
            flux_in_dim=flux_in_dim,
            background_diffusion=background_diffusion,
        )

    elif kind == "advection_diffusion":
        v = torch.tensor(velocity, dtype=x.dtype)
        rhs = advection_diffusion_rhs(
            x,
            kappa=kappa,
            velocity=v,
            freq_x=freq_x,
            freq_y=freq_y,
        )

        mu_vector = torch.zeros(x.shape[0], 2, dtype=x.dtype)
        mu_vector[:, 0] = v[0]
        mu_vector[:, 1] = v[1]
        mu_scalar = None

        flux_in_dim = 2 + 4
        background_diffusion = kappa

        return ManufacturedProblem(
            name="advection_diffusion",
            rhs=rhs,
            boundary_values=boundary_values,
            target_u=target_u,
            mu_scalar=mu_scalar,
            mu_vector=mu_vector,
            flux_in_dim=flux_in_dim,
            background_diffusion=background_diffusion,
        )

    else:
        raise ValueError(f"Unknown problem kind: {kind}")


# ============================================================
# Main experiment driver
# ============================================================

def run_experiment(
    kind: str = "poisson",
    nx: int = 21,
    ny: int = 21,
    eps_graph: float = 0.17,
    hidden_dim: int = 64,
    depth: int = 4,
    train_steps: int = 500,
    lr: float = 1e-3,
    kappa: float = 0.1,
    velocity=(1.0, 0.5),
    freq_x: int = 1,
    freq_y: int = 1,
    train_device: str = "cuda",
    plot_filename: str | None = None,
):
    torch.set_default_dtype(torch.float64)

    # --------------------------------------------------------
    # Geometry / graph / MEEC
    # --------------------------------------------------------
    x = make_square_points(nx=nx, ny=ny)
    boundary_mask = make_boundary_mask(x)

    graph = build_epsilon_ball_graph(x, eps_graph)
    meec = build_meec_complex(
        graph=graph,
        boundary_mask=boundary_mask,
        domain_volume=1.0,
        eps=eps_graph,
    )

    # --------------------------------------------------------
    # Manufactured problem
    # --------------------------------------------------------
    problem = build_problem(
        x=x,
        boundary_mask=boundary_mask,
        kind=kind,
        kappa=kappa,
        velocity=velocity,
        freq_x=freq_x,
        freq_y=freq_y,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------
    if kind == "poisson":
        model = MEECNet(
            flux_net=ZeroFluxNet(),
            background_diffusion=problem.background_diffusion,
        )
    else:
        model = MEECNet(
            flux_net=FluxMLP(
                in_dim=problem.flux_in_dim,
                hidden_dim=hidden_dim,
                depth=depth,
            ),
            background_diffusion=problem.background_diffusion,
        )

    dataset = [
        Sample(
            meec=meec,
            rhs=problem.rhs,
            boundary_values=problem.boundary_values,
            target_u=problem.target_u,
            mu_scalar=problem.mu_scalar,
            mu_vector=problem.mu_vector,
        )
    ]

    print(f"Problem: {problem.name}")
    print(f"Nodes: {x.shape[0]}, Edges: {graph.senders.numel()}")
    print(f"Manufactured frequencies: freq_x={freq_x}, freq_y={freq_y}")

    # --------------------------------------------------------
    # Initial / pre-training solve
    # --------------------------------------------------------
    with torch.no_grad():
        pred0 = solve_dirichlet_nonlinear(
            model=model,
            meec=meec,
            rhs=problem.rhs,
            boundary_values=problem.boundary_values,
            mu_scalar=problem.mu_scalar,
            mu_vector=problem.mu_vector,
            verbose=True,
        )
        rel0 = torch.linalg.norm(pred0 - problem.target_u) / torch.clamp(
            torch.linalg.norm(problem.target_u), min=1e-12
        )
        mse0 = torch.mean((pred0 - problem.target_u) ** 2)

    print(f"Initial relative L2 error: {float(rel0):.6e}")
    print(f"Initial MSE: {float(mse0):.6e}")

    # --------------------------------------------------------
    # Train only if not Poisson
    # --------------------------------------------------------
    if kind != "poisson":
        model = train(
            model=model,
            dataset=dataset,
            num_steps=train_steps,
            lr=lr,
            grad_clip=1.0,
            print_every=max(1, train_steps // 100),
            device=train_device,
        )
        model.to("cpu")
    else:
        print("Skipping training for Poisson: validating MEEC/background diffusion only.")

    # --------------------------------------------------------
    # Final solve
    # --------------------------------------------------------
    with torch.no_grad():
        pred = solve_dirichlet_nonlinear(
            model=model,
            meec=meec,
            rhs=problem.rhs,
            boundary_values=problem.boundary_values,
            mu_scalar=problem.mu_scalar,
            mu_vector=problem.mu_vector,
        )
        rel = torch.linalg.norm(pred - problem.target_u) / torch.clamp(
            torch.linalg.norm(problem.target_u), min=1e-12
        )
        mse = torch.mean((pred - problem.target_u) ** 2)

    print(f"Final relative L2 error: {float(rel):.6e}")
    print(f"Final MSE: {float(mse):.6e}")

    if plot_filename is None:
        plot_filename = f"outputs/{kind}_fx{freq_x}_fy{freq_y}_solution.pdf"

    save_solution_plot(
        x=x,
        u_pred=pred,
        u_true=problem.target_u,
        filename=plot_filename,
        title=f"{kind}: predicted vs true solution (fx={freq_x}, fy={freq_y})",
    )
    print(f"Saved plot to: {plot_filename}")

    return {
        "x": x,
        "boundary_mask": boundary_mask,
        "meec": meec,
        "problem": problem,
        "model": model,
        "pred": pred,
        "pred0": pred0,
        "rel_error": rel,
        "mse": mse,
        "plot_filename": plot_filename,
    }


if __name__ == "__main__":
    # Flip this between:
    #   "poisson"
    #   "advection_diffusion"
    kind = "poisson"

    out = run_experiment(
        kind=kind,
        nx=51,
        ny=51,
        eps_graph=0.05,
        hidden_dim=64,
        depth=3,
        train_steps=500,
        lr=1e-3,
        kappa=0.1,
        velocity=(1.0, 0.5),
        freq_x=2,
        freq_y=3,
        train_device="cuda",
        plot_filename=f"outputs/{kind}_solution.pdf",
    )
