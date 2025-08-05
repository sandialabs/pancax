from .base import AbstractPancaxModel
from jaxtyping import Array, Float
from typing import Optional
import equinox as eqx
import jax
import jax.numpy as jnp
import math


class Graph(eqx.Module):
    node_features: Float[Array, "nn nf"]
    # TODO eventually try out sparse matrices
    edge_indices: Float[Array, "2 ne"]


class GraphLinear(AbstractPancaxModel):
    # parameters
    bias: Float[Array, "no"]
    weight: Float[Array, "no ni"]

    # statics
    use_bias: bool = eqx.field(static=True)

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        use_bias: Optional[bool] = True,
        *,
        key: jax.random.PRNGKey
    ):
        self.use_bias = use_bias

        wkey, bkey = jax.random.split(key, 2)
        lim = 1 / math.sqrt(n_inputs)
        self.bias = jax.random.uniform(
            key, (n_outputs,), minval=-lim, maxval=lim
        )
        self.weight = jax.random.uniform(
            key, (n_outputs, n_inputs), minval=-lim, maxval=lim
        )

    def __call__(self, x: Graph) -> Graph:
        y = self.weight @ x.node_features.T
        if self.use_bias:
            y = y.T + self.bias
        return Graph(y, x.edge_indices)


class GCNLayer(AbstractPancaxModel):
    projection: GraphLinear

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        *,
        key: jax.random.PRNGKey
    ):
        self.projection = GraphLinear(n_inputs, n_outputs, key=key)

    def __call__(self, graph: Graph):
        x = self.projection(graph)  # shape [num_nodes, out_dim]
        
        # Basic message passing: sum over neighbors
        src, dst = graph.edge_indices

        messages = x.node_features[src]
        aggregated = jax.ops.segment_sum(messages, dst, num_segments=x.node_features.shape[0])
        
        return Graph(aggregated, graph.edge_indices)  # [num_nodes, out_dim]