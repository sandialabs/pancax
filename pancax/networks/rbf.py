import equinox as eqx
import jax
import jax.numpy as jnp


class RBFBasis(eqx.Module):
    center: jax.Array
    sigma: jax.Array

    def __init__(
        self,
        n_inputs: int,
        n_neurons: int,
        key: jax.random.PRNGKey
    ) -> None:
        self.center = jax.random.normal(key, (n_neurons, n_inputs))
        self.sigma = 1.0 * jnp.ones((n_neurons,))

    # make an input for different rbf funcs
    def rbf_func(self, x, center, sigma):
        return jnp.exp(-jnp.linalg.norm(x - center)**2 / sigma)

    def __call__(self, x):
        rbfs = jax.vmap(
            self.rbf_func, in_axes=(None, 0, 0)
        )(x, self.center, self.sigma)
        return rbfs

    # def __call__(self, x):
    #     out = self.center - x
    #     out = jax.vmap(lambda a: jnp.dot(a, a))(out)
    #     out = jnp.exp(-out / jnp.square(self.sigma))
    #     # return rbf_normalization(out)
    #     return out


# def rbf_normalization(hidden):
#     # return hidden / jnp.sum(hidden, axis=1)[:, None]
#     return hidden / jnp.sum(hidden)[:, None]
