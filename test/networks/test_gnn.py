import pytest


@pytest.fixture
def gcn_layer():
    from pancax.networks.gnn import GCNLayer
    import jax.random as random
    return GCNLayer(2, 5, key=random.PRNGKey(0))


@pytest.fixture
def graphs():
    from pancax.networks.gnn import Graph
    import jax.numpy as jnp
    node_feats = jnp.arange(8).reshape((4, 2))
    # adj_matrix = jnp.array([
    #     [1, 1, 0, 0],
    #     [1, 1, 1, 1],
    #     [0, 1, 1, 1],
    #     [0, 1, 1, 1]
    # ])
    src = jnp.array([0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    dst = jnp.array([0, 1, 0, 1, 2, 3, 1, 2, 3, 1, 2, 3])
    edge_indices = jnp.stack([src, dst])
    return [Graph(node_feats, edge_indices)]


def test_gcn_layer(gcn_layer, graphs):
    y = gcn_layer(graphs[0])
    print(graphs[0])
    print(gcn_layer)
    print(y)

    assert False