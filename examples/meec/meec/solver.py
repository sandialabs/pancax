import torch
from torch.func import jacrev

from .complex import MEECComplex
from .model import MEECNet


# ============================================================
# Basic operators
# ============================================================

def apply_d0(u: torch.Tensor, meec: MEECComplex) -> torch.Tensor:
    return meec.d0 @ u


def apply_laplacian(u: torch.Tensor, meec: MEECComplex) -> torch.Tensor:
    du = apply_d0(u, meec)
    return meec.d0.T @ (meec.m1 * du)


def apply_div(edge_cochain: torch.Tensor, meec: MEECComplex) -> torch.Tensor:
    return meec.d0.T @ (meec.m1 * edge_cochain)


# ============================================================
# Residual
# ============================================================

def residual(
    model: MEECNet,
    u: torch.Tensor,
    meec: MEECComplex,
    rhs: torch.Tensor,
    boundary_values: torch.Tensor,
    mu_scalar: torch.Tensor | None = None,
    mu_vector: torch.Tensor | None = None,
    forcing_features: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Residual:
        eps * d0^T M1 d0 u + d0^T M1 F_theta(u;mu) - M0 rhs = 0
    with strong Dirichlet enforcement by replacing boundary rows with u-g.
    """
    u_full = torch.where(meec.boundary_mask, boundary_values, u)

    diff_term = model.background_diffusion * apply_laplacian(u_full, meec)
    flux_term = apply_div(
        model.edge_flux(
            u=u_full,
            meec=meec,
            mu_scalar=mu_scalar,
            mu_vector=mu_vector,
            forcing=forcing_features,
        ),
        meec,
    )

    res = diff_term + flux_term - meec.m0 * rhs
    res = torch.where(meec.boundary_mask, u_full - boundary_values, res)
    return res


# ============================================================
# Jacobian wrt state using torch.func.jacrev
# ============================================================

def jacobian_wrt_u(
    model: MEECNet,
    u: torch.Tensor,
    meec: MEECComplex,
    rhs: torch.Tensor,
    boundary_values: torch.Tensor,
    mu_scalar: torch.Tensor | None = None,
    mu_vector: torch.Tensor | None = None,
    forcing_features: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute residual and dense Jacobian dR/du using torch.func.jacrev.

    Returns:
        r: (N,)
        J: (N, N)
    """
    def rfun(uu: torch.Tensor) -> torch.Tensor:
        return residual(
            model=model,
            u=uu,
            meec=meec,
            rhs=rhs,
            boundary_values=boundary_values,
            mu_scalar=mu_scalar,
            mu_vector=mu_vector,
            forcing_features=forcing_features,
        )

    # For jacrev, `u` should require grad logically, but we do not need to
    # manually call requires_grad_ here.
    r = rfun(u)
    J = jacrev(rfun)(u)
    return r, J


# ============================================================
# Forward nonlinear solve
# ============================================================

def solve_dirichlet_nonlinear(
    model: MEECNet,
    meec: MEECComplex,
    rhs: torch.Tensor,
    boundary_values: torch.Tensor,
    mu_scalar: torch.Tensor | None = None,
    mu_vector: torch.Tensor | None = None,
    forcing_features: torch.Tensor | None = None,
    u0: torch.Tensor | None = None,
    max_newton: int = 8,
    tol: float = 1e-6,
    line_search_steps: int = 8,
    verbose: bool = False,
) -> torch.Tensor:
    """
    Forward-only Newton solve.
    This is intentionally NOT differentiated through.
    """
    if u0 is None:
        u = boundary_values.detach().clone()
    else:
        u = u0.detach().clone()

    u = torch.where(meec.boundary_mask, boundary_values, u)

    for k in range(max_newton):
        r, J = jacobian_wrt_u(
            model=model,
            u=u,
            meec=meec,
            rhs=rhs,
            boundary_values=boundary_values,
            mu_scalar=mu_scalar,
            mu_vector=mu_vector,
            forcing_features=forcing_features,
        )

        rnorm = torch.linalg.norm(r).detach()
        if verbose:
            print(f"    Newton iter {k+1}: residual = {float(rnorm):.6e}")

        if float(rnorm) < tol:
            break

        N = u.numel()
        I = torch.eye(N, dtype=J.dtype, device=J.device)
        du = torch.linalg.solve(J + 1e-8 * I, -r)

        accepted = False
        alpha = 1.0
        for ls in range(line_search_steps):
            u_trial = u + alpha * du
            u_trial = torch.where(meec.boundary_mask, boundary_values, u_trial)

            r_trial = residual(
                model=model,
                u=u_trial,
                meec=meec,
                rhs=rhs,
                boundary_values=boundary_values,
                mu_scalar=mu_scalar,
                mu_vector=mu_vector,
                forcing_features=forcing_features,
            )
            trial_norm = torch.linalg.norm(r_trial).detach()

            if verbose:
                print(
                    f"      line search {ls+1}: alpha={alpha:.3e}, "
                    f"norm={float(trial_norm):.6e}"
                )

            if float(trial_norm) < float(rnorm):
                u = u_trial.detach()
                accepted = True
                break
            alpha *= 0.5

        if not accepted:
            if verbose:
                print("    Newton line search failed.")
            break

    u = torch.where(meec.boundary_mask, boundary_values, u)
    return u.detach()


# ============================================================
# Adjoint-style gradient computation
# ============================================================

def adjoint_parameter_grads(
    model: MEECNet,
    u_star: torch.Tensor,
    target_u: torch.Tensor,
    meec: MEECComplex,
    rhs: torch.Tensor,
    boundary_values: torch.Tensor,
    mu_scalar: torch.Tensor | None = None,
    mu_vector: torch.Tensor | None = None,
    forcing_features: torch.Tensor | None = None,
    loss_reduction: str = "mean",
):
    """
    Compute parameter gradients using the implicit/adjoint formula.

    Loss:
        L = mean((u_star - target_u)^2)
    """
    diff = u_star - target_u

    if loss_reduction == "mean":
        loss = torch.mean(diff ** 2)
        dL_du = (2.0 / diff.numel()) * diff
    elif loss_reduction == "sum":
        loss = torch.sum(diff ** 2)
        dL_du = 2.0 * diff
    else:
        raise ValueError(f"Unknown loss_reduction={loss_reduction}")

    # Rebuild residual graph at converged state
    u_req = u_star.detach().clone().requires_grad_(True)

    params = [p for p in model.parameters() if p.requires_grad]

    def rfun(uu: torch.Tensor) -> torch.Tensor:
        return residual(
            model=model,
            u=uu,
            meec=meec,
            rhs=rhs,
            boundary_values=boundary_values,
            mu_scalar=mu_scalar,
            mu_vector=mu_vector,
            forcing_features=forcing_features,
        )

    r = rfun(u_req)
    J = jacrev(rfun)(u_req)

    N = u_req.numel()
    I = torch.eye(N, dtype=J.dtype, device=J.device)

    # Solve adjoint:
    #   J^T lambda = dL/du
    lam = torch.linalg.solve(J.T + 1e-8 * I, dL_du)

    # Parameter gradients:
    #   dL/dtheta = - lam^T (dR/dtheta)
    grads = torch.autograd.grad(
        outputs=r,
        inputs=params,
        grad_outputs=-lam,
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )

    grads = [
        g if g is not None else torch.zeros_like(p)
        for g, p in zip(grads, params)
    ]

    return loss.detach(), grads


def assign_parameter_grads(model: MEECNet, grads, accumulate: bool = True):
    params = [p for p in model.parameters() if p.requires_grad]
    for p, g in zip(params, grads):
        if p.grad is None:
            p.grad = g.detach().clone()
        else:
            if accumulate:
                p.grad.add_(g.detach())
            else:
                p.grad.zero_()
                p.grad.add_(g.detach())
