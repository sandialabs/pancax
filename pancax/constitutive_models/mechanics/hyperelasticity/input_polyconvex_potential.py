from ..base import HyperelasticModel
from ....networks.input_polyconvex_nn import InputPolyconvexNN
from ...properties import Property
import equinox as eqx
import jax
import jax.numpy as jnp


class InputPolyConvexPotential(HyperelasticModel):
    bulk_modulus: Property
    network: InputPolyconvexNN

    def __init__(
        self,
        bulk_modulus: Property,
        key: jax.random.PRNGKey
    ):
        self.bulk_modulus = bulk_modulus
        self.network = InputPolyconvexNN(
            n_convex=2,
            n_inputs=2,
            n_outputs=1,
            activation_x=jax.nn.softplus,
            activation_y=jax.nn.softplus,
            key=key
        )

    def energy(self, grad_u, theta, state_old, dt):
        K = self.bulk_modulus
        J = self.jacobian(grad_u)
        I1_bar = self.I1_bar(grad_u)
        I2_bar = self.I2_bar(grad_u)
        invariants = jnp.array([I1_bar, I2_bar])

        W_vol = 0.5 * K * (0.5 * (J**2 - 1) - jnp.log(J))
        W_dev_nn = self.network(invariants)
        W_dev_0 = self.network(jnp.array([3., 3.]))
        # W_dev_s = 2. * jnp.sum(jax.grad(self.network))(invariants) * (J - 1.)
        grad_W_dev_nn = jax.grad(lambda x: self.network(x)[0])(
            jnp.array([3., 3.])
        )
        # TODO probably need to add dpsi/dJ to the term below
        W_dev_s = 2. * (grad_W_dev_nn[0] + 2. * grad_W_dev_nn[1]) * (J - 1.)

        return W_vol + (W_dev_nn - W_dev_0 - W_dev_s), state_old

    def parameter_enforcement(self):
        self = eqx.tree_at(
            lambda x: x.network,
            self,
            self.network.parameter_enforcement()
        )
        return self
