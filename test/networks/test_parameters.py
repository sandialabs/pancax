# from pancax import Parameters
# import jax


# # def test_fixed_properties():
# #     props = FixedParameters([1., 2.])
# #     props = props()
# #     assert jax.numpy.array_equal(props, jax.numpy.array([1., 2.]))


# def test_properties_bounds():
#     props = Parameters(
#         prop_mins=[1., 2.],
#         prop_maxs=[2., 3.],
#         key=jax.random.key(0)
#     )
#     props = props()
#     assert props[0] >= 1. and props[0] <= 2.    
#     assert props[1] >= 2. and props[1] <= 3.
