from dataclasses import dataclass
import torch


@dataclass
class EpsilonBallGraph:
    x: torch.Tensor              # (N, 2)
    senders: torch.Tensor        # (E,)
    receivers: torch.Tensor      # (E,)
    dx: torch.Tensor             # (E, 2)
    r: torch.Tensor              # (E,)
    midpoints: torch.Tensor      # (E, 2)
    tangents: torch.Tensor       # (E, 2)


def build_epsilon_ball_graph(x: torch.Tensor, eps: float) -> EpsilonBallGraph:
    """
    Build canonical undirected epsilon-ball graph with edges i<j.
    """
    assert x.ndim == 2 and x.shape[1] == 2
    N = x.shape[0]

    diff = x[None, :, :] - x[:, None, :]   # x_j - x_i at [i,j]
    dist = torch.linalg.norm(diff, dim=-1)

    ii, jj = torch.triu_indices(N, N, offset=1)
    mask = dist[ii, jj] < eps

    senders = ii[mask]
    receivers = jj[mask]

    dx = x[receivers] - x[senders]
    r = torch.linalg.norm(dx, dim=-1)
    tangents = dx / torch.clamp(r[:, None], min=1e-12)
    midpoints = 0.5 * (x[senders] + x[receivers])

    return EpsilonBallGraph(
        x=x,
        senders=senders,
        receivers=receivers,
        dx=dx,
        r=r,
        midpoints=midpoints,
        tangents=tangents,
    )
