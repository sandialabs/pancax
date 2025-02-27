# from pancax import FieldPhysicsPair, Parameters, MLP
# from pathlib import Path
# import equinox as eqx
# import jax
# import os


# def test_field_property_pair():
#     network = MLP(3, 2, 20, 3, jax.nn.tanh, key=jax.random.key(0))
#     props = Parameters(
#         prop_mins=[1., 2.],
#         prop_maxs=[2., 3.],
#         key=jax.random.key(0)
#     )
#     model = FieldPhysicsPair(network, props)
#     x = jax.numpy.ones(3)

#     network, props = model
#     y = network(x)
#     props = props()
#     assert y.shape == (2,)
#     assert props[0] >= 1. and props[0] <= 2.    
#     assert props[1] >= 2. and props[1] <= 3.


# def test_model_serialisation():
#     network = MLP(3, 2, 20, 3, jax.nn.tanh, key=jax.random.key(0))
#     props = Parameters(
#         prop_mins=[1., 2.],
#         prop_maxs=[2., 3.],
#         key=jax.random.key(0)
#     )
#     model = FieldPhysicsPair(network, props)
#     x = jax.numpy.ones(3)

#     network, props = model
#     y_old = network(x)
#     props_old = props()

#     model.serialise(os.path.join(Path(__file__).parent, 'checkpoint'), 0)

#     model_loaded = eqx.tree_deserialise_leaves(
#         os.path.join(Path(__file__).parent, 'checkpoint_0000000.eqx'), 
#         model
#     )
#     network, props = model_loaded
#     y_new = network(x)
#     props_new = props()
#     assert jax.numpy.array_equal(y_old, y_new)
#     assert jax.numpy.array_equal(props_old, props_new)
