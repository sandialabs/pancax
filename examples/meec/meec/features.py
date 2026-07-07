import torch
from .complex import MEECComplex


def edge_scalar_features(u: torch.Tensor, meec: MEECComplex) -> torch.Tensor:
    ui = u[meec.senders]
    uj = u[meec.receivers]
    ubar = 0.5 * (ui + uj)
    du_dn = (uj - ui) / torch.clamp(meec.r, min=1e-12)
    return torch.stack([ubar, du_dn], dim=-1)


def edge_vector_projection_features(v: torch.Tensor, meec: MEECComplex) -> torch.Tensor:
    vi = v[meec.senders]
    vj = v[meec.receivers]
    vbar = 0.5 * (vi + vj)
    dv = (vj - vi) / torch.clamp(meec.r[:, None], min=1e-12)

    t = meec.tangents
    n = torch.stack([-t[:, 1], t[:, 0]], dim=-1)

    return torch.stack(
        [
            (vbar * t).sum(dim=-1),
            (vbar * n).sum(dim=-1),
            (dv * t).sum(dim=-1),
            (dv * n).sum(dim=-1),
        ],
        dim=-1,
    )


def make_edge_features(
    u: torch.Tensor,
    meec: MEECComplex,
    mu_scalar: torch.Tensor | None = None,
    mu_vector: torch.Tensor | None = None,
    forcing: torch.Tensor | None = None,
) -> torch.Tensor:
    feats = [edge_scalar_features(u, meec)]

    if mu_scalar is not None:
        feats.append(edge_scalar_features(mu_scalar, meec))

    if mu_vector is not None:
        feats.append(edge_vector_projection_features(mu_vector, meec))

    if forcing is not None:
        feats.append(edge_scalar_features(forcing, meec))

    return torch.cat(feats, dim=-1)
