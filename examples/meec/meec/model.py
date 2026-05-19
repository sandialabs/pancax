import torch
import torch.nn as nn

from .complex import MEECComplex
from .features import make_edge_features


class FluxMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, depth: int = 4):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.Tanh())
            d = hidden_dim
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, edge_features: torch.Tensor) -> torch.Tensor:
        return self.net(edge_features).squeeze(-1)


class MEECNet(nn.Module):
    def __init__(self, flux_net: FluxMLP, background_diffusion: float = 1e-2):
        super().__init__()
        self.flux_net = flux_net
        self.background_diffusion = background_diffusion

    def forward(
        self,
        u: torch.Tensor,
        meec: MEECComplex,
        mu_scalar: torch.Tensor | None = None,
        mu_vector: torch.Tensor | None = None,
        forcing: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Returns edge flux cochain F_theta(u;mu).
        """
        xi = make_edge_features(
            u=u,
            meec=meec,
            mu_scalar=mu_scalar,
            mu_vector=mu_vector,
            forcing=forcing,
        )
        density = self.flux_net(xi)
        return meec.r * density

    def edge_flux(
        self,
        u: torch.Tensor,
        meec: MEECComplex,
        mu_scalar: torch.Tensor | None = None,
        mu_vector: torch.Tensor | None = None,
        forcing: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.forward(
            u=u,
            meec=meec,
            mu_scalar=mu_scalar,
            mu_vector=mu_vector,
            forcing=forcing,
        )
