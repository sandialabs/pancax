from .base import AbstractPancaxModel
from jaxtyping import Array, Float
from pancax.distance_functions import \
    distance_function, make_dirichlet_extension
import jax
import jax.numpy as jnp


class _TrialFunction(AbstractPancaxModel):
    dfs: any
    gs: any
    n_dofs: int

    def __init__(self, problem) -> None:
        self.n_dofs = problem.physics.n_dofs
        dbc_ssets_by_dof = [[] for _ in range(self.n_dofs)]
        dbcs_by_dof = [[] for _ in range(self.n_dofs)]
        for bc in problem.dirichlet_bcs:
            dbc_ssets_by_dof[bc.component].append(bc.sset_name)
            dbcs_by_dof[bc.component].append(bc)

        self.dfs = [
            distance_function(problem.domain, ssets)
            for ssets in dbc_ssets_by_dof
        ]
        self.gs = [
            make_dirichlet_extension(problem.domain, bcs)
            for bcs in dbcs_by_dof
        ]

    def __call__(self, x, t, z):
        u = jnp.zeros(self.n_dofs)
        for n in range(self.n_dofs):
            # this will potentially do something stupid and
            # enforce zero ics even if you provide
            # explicit ics
            u_n = self.gs[n](x, t) + t * self.dfs[n](x) * z[n]
            u = u.at[n].set(u_n)
        return u


class TrialFunction(AbstractPancaxModel):
    model: AbstractPancaxModel
    normalize_time: bool  # to help with static problems
    tf: any
    t_min: Float[Array, "1"]
    t_max: Float[Array, "1"]
    x_mins: Float[Array, "nd"]  # = jnp.zeros(3)
    x_maxs: Float[Array, "nd"]  # = jnp.zeros(3)

    def __init__(self, problem, model) -> None:
        self.model = model
        self.tf = _TrialFunction(problem)
        self.t_min = jnp.min(problem.times, axis=0)
        self.t_max = jnp.max(problem.times, axis=0)
        self.x_mins = jnp.min(problem.coords, axis=0)
        self.x_maxs = jnp.max(problem.coords, axis=0)

        if jnp.allclose(self.t_min, self.t_max):
            self.normalize_time = False
        else:
            self.normalize_time = True

    def __call__(self, x, t):
        x_norm = (x - self.x_mins) / (self.x_maxs - self.x_mins)
        t_norm = jax.lax.cond(
            self.normalize_time,
            lambda z: (z - self.t_min) / (self.t_max - self.t_min),
            lambda z: z,
            t
        )
        inputs = jnp.hstack((x_norm, t_norm))
        z = self.model(inputs)
        return self.tf(x, t, z)
