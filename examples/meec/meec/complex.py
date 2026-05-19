from dataclasses import dataclass
from .graph import EpsilonBallGraph
import numpy as np
import torch
import scipy.sparse as sp
import scipy.sparse.linalg as spla



@dataclass
class MEECComplex:
    x: torch.Tensor
    senders: torch.Tensor
    receivers: torch.Tensor
    d0: torch.Tensor              # (E, N)
    m0: torch.Tensor              # (N,)
    m1: torch.Tensor              # (E,)
    interior_mask: torch.Tensor   # (N,) bool
    boundary_mask: torch.Tensor   # (N,) bool
    dx: torch.Tensor              # (E, 2)
    r: torch.Tensor               # (E,)
    midpoints: torch.Tensor       # (E, 2)
    tangents: torch.Tensor        # (E, 2)


def wendland_like_kernel(r: torch.Tensor, eps: float) -> torch.Tensor:
    z = 1.0 - r / eps
    return torch.where(z > 0.0, z**2, torch.zeros_like(z))


def build_incidence_matrix(num_nodes: int, senders: torch.Tensor, receivers: torch.Tensor) -> torch.Tensor:
    E = senders.numel()
    d0 = torch.zeros(E, num_nodes, dtype=torch.float64)
    d0[torch.arange(E), senders] = -1.0
    d0[torch.arange(E), receivers] = +1.0
    return d0


def build_virtual_volumes(
    x: torch.Tensor,
    senders: torch.Tensor,
    receivers: torch.Tensor,
    r: torch.Tensor,
    eps: float,
    domain_volume: float,
    interior_mask: torch.Tensor,
) -> torch.Tensor:
    N = x.shape[0]
    phi_e = wendland_like_kernel(r, eps)

    kappa = torch.zeros(N, dtype=torch.float64)
    kappa.index_add_(0, senders, phi_e)
    kappa.index_add_(0, receivers, phi_e)

    inv_kappa = torch.zeros_like(kappa)
    inv_kappa[interior_mask] = 1.0 / torch.clamp(kappa[interior_mask], min=1e-12)
    inv_sum = inv_kappa.sum().clamp(min=1e-12)

    m = torch.zeros_like(kappa)
    m[interior_mask] = domain_volume * inv_kappa[interior_mask] / inv_sum
    return m


def _incident_edges(node_idx: int, senders: np.ndarray, receivers: np.ndarray):
    inc_sender = senders == node_idx
    inc_receiver = receivers == node_idx
    inc = inc_sender | inc_receiver
    edge_idx = np.where(inc)[0]
    return edge_idx, inc_sender, inc_receiver


def solve_edge_areas_moment_matching(
    x: torch.Tensor,
    senders_t: torch.Tensor,
    receivers_t: torch.Tensor,
    dx_t: torch.Tensor,
    r_t: torch.Tensor,
    m0_t: torch.Tensor,
    interior_mask_t: torch.Tensor,
    eps: float,
    ridge: float = 1e-10,
) -> torch.Tensor:
    """
    Solve:
      min 1/2 a^T Phi^{-1} a  s.t. B a = c
    using Schur complement:
      a = Phi B^T (B Phi B^T)^(-1) c

    Implemented with SciPy sparse linear algebra on CPU, matching the style described in the paper.
    """
    senders = senders_t.detach().cpu().numpy()
    receivers = receivers_t.detach().cpu().numpy()
    dx = dx_t.detach().cpu().numpy()
    r = r_t.detach().cpu().numpy()
    m0 = m0_t.detach().cpu().numpy()
    interior_mask = interior_mask_t.detach().cpu().numpy()

    N = x.shape[0]
    E = senders.shape[0]

    phi = np.maximum(1.0 - r / eps, 0.0) ** 2
    phi = phi + 1e-12

    rows = []
    cols = []
    vals = []
    rhs = []
    row_counter = 0

    for i in range(N):
        if not interior_mask[i]:
            continue

        edge_idx, inc_sender, inc_receiver = _incident_edges(i, senders, receivers)
        if edge_idx.size == 0:
            continue

        # eta points from node i to the other endpoint
        eta = np.zeros((edge_idx.size, 2), dtype=np.float64)
        for k, e in enumerate(edge_idx):
            if senders[e] == i:
                eta[k] = dx[e]
            else:
                eta[k] = -dx[e]

        # 5 constraints in 2D:
        # sum a eta_x = 0
        # sum a eta_y = 0
        # sum a eta_x^2 = 2 m_i
        # sum a eta_x eta_y = 0
        # sum a eta_y^2 = 2 m_i
        local_rows = [
            eta[:, 0],
            eta[:, 1],
            eta[:, 0] * eta[:, 0],
            eta[:, 0] * eta[:, 1],
            eta[:, 1] * eta[:, 1],
        ]
        local_rhs = [0.0, 0.0, 2.0 * m0[i], 0.0, 2.0 * m0[i]]

        for rr, b in zip(local_rows, local_rhs):
            rows.extend([row_counter] * edge_idx.size)
            cols.extend(edge_idx.tolist())
            vals.extend(rr.tolist())
            rhs.append(b)
            row_counter += 1

    C = row_counter
    if C == 0:
        return torch.ones(E, dtype=torch.float64)

    B = sp.coo_matrix((vals, (rows, cols)), shape=(C, E)).tocsr()
    Phi = sp.diags(phi, offsets=0, format="csr")
    S = (B @ Phi @ B.T).tocsr()
    S = S + ridge * sp.eye(C, format="csr")

    lu = spla.splu(S.tocsc())
    lam = lu.solve(np.asarray(rhs, dtype=np.float64))
    a = Phi @ (B.T @ lam)

    return torch.tensor(np.asarray(a), dtype=torch.float64)


def build_meec_complex(
    graph: EpsilonBallGraph,
    boundary_mask: torch.Tensor,
    domain_volume: float,
    eps: float,
    ridge: float = 1e-10,
) -> MEECComplex:
    x = graph.x.to(dtype=torch.float64)
    senders = graph.senders
    receivers = graph.receivers
    boundary_mask = boundary_mask.bool()
    interior_mask = ~boundary_mask

    d0 = build_incidence_matrix(x.shape[0], senders, receivers)
    m0 = build_virtual_volumes(
        x=x,
        senders=senders,
        receivers=receivers,
        r=graph.r.to(dtype=torch.float64),
        eps=eps,
        domain_volume=domain_volume,
        interior_mask=interior_mask,
    )
    m1 = solve_edge_areas_moment_matching(
        x=x,
        senders_t=senders,
        receivers_t=receivers,
        dx_t=graph.dx.to(dtype=torch.float64),
        r_t=graph.r.to(dtype=torch.float64),
        m0_t=m0,
        interior_mask_t=interior_mask,
        eps=eps,
        ridge=ridge,
    )

    return MEECComplex(
        x=x,
        senders=senders,
        receivers=receivers,
        d0=d0,
        m0=m0,
        m1=m1,
        interior_mask=interior_mask,
        boundary_mask=boundary_mask,
        dx=graph.dx.to(dtype=torch.float64),
        r=graph.r.to(dtype=torch.float64),
        midpoints=graph.midpoints.to(dtype=torch.float64),
        tangents=graph.tangents.to(dtype=torch.float64),
    )

def complex_to(meec: MEECComplex, device: torch.device | str) -> MEECComplex:
    return MEECComplex(
        x=meec.x.to(device),
        senders=meec.senders.to(device),
        receivers=meec.receivers.to(device),
        d0=meec.d0.to(device),
        m0=meec.m0.to(device),
        m1=meec.m1.to(device),
        interior_mask=meec.interior_mask.to(device),
        boundary_mask=meec.boundary_mask.to(device),
        dx=meec.dx.to(device),
        r=meec.r.to(device),
        midpoints=meec.midpoints.to(device),
        tangents=meec.tangents.to(device),
    )
