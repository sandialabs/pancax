from ..base import HyperelasticModel
from ....networks.input_polyconvex_nn import InputPolyconvexNN
from ...properties import Property
import equinox as eqx
import jax
import jax.numpy as jnp


class InputPolyConvexPotential(HyperelasticModel):
    # parameters
    bulk_modulus: Property
    beta: float
    gamma: float
    zeta: float
    log_alpha: eqx.field(static=True)
    use_l0_regularization: bool
    # network
    network: InputPolyconvexNN

    def __init__(
        self,
        bulk_modulus: Property,
        key: jax.random.PRNGKey,
        beta: float = 2.0 / 3.0,
        gamma: float = -0.1,
        zeta: float = 1.1,
        use_l0_regularization: bool = False
    ):
        self.bulk_modulus = bulk_modulus
        self.beta = beta
        self.gamma = gamma
        self.zeta = zeta
        self.use_l0_regularization = use_l0_regularization

        if len(key.shape) == 1:
            self.network = InputPolyconvexNN(
                n_convex=3,
                n_inputs=3,
                n_outputs=1,
                activation_x=jax.nn.softplus,
                activation_y=jax.nn.softplus,
                key=key
            )

            def alpha_func(p):
                return 0.01 * jax.random.normal(key, p.shape)

            self.log_alpha = jax.tree.map(
                lambda x: alpha_func(x),
                eqx.partition(self.network, eqx.is_array)[0]
            )

        elif len(key.shape) == 2:
            @eqx.filter_vmap
            def vmap_func(_key):
                return InputPolyconvexNN(
                    n_convex=3,
                    n_inputs=3,
                    n_outputs=1,
                    activation_x=jax.nn.softplus,
                    activation_y=jax.nn.softplus,
                    key=_key
                )

            self.network = vmap_func(key)
        else:
            raise ValueError(
                f"Invalid shape for key {key} with shape {key.shape}"
            )

    def energy(self, grad_u, theta, state_old, dt, *args):
        gate_key = args[0]

        # K = self.bulk_modulus
        J = self.jacobian(grad_u)
        I1_bar = self.I1_bar(grad_u)
        I2_bar = self.I2_bar(grad_u)
        invariants = jnp.array([I1_bar, I2_bar, J])

        # gate_params = self.l0_regularization_gate(gate_key)
        if self.use_l0_regularization:
            gate_params = self.l0_regularization_gate(gate_key)
        else:
            gate_params = self.network

        # W_vol = 0.5 * K * (0.5 * (J**2 - 1) - jnp.log(J))

        # W_dev_s = 2. * jnp.sum(jax.grad(self.network))(invariants) * (J - 1.)
        # grad_W_dev_nn = jax.grad(lambda x: self.network(x)[0])(
        #     jnp.array([3., 3.])
        # )
        # W_dev_nn = self.network(invariants)
        # W_dev_0 = self.network(jnp.array([3., 3., 1.]))
        # grad_W_dev_nn = jax.grad(lambda x: self.network(x)[0])(
        #     jnp.array([3., 3., 1.])
        # )
        W_dev_nn = gate_params(invariants)
        W_dev_0 = gate_params(jnp.array([3., 3., 1.]))
        grad_W_dev_nn = jax.grad(lambda x: gate_params(x)[0])(
            jnp.array([3., 3., 1.])
        )
        # TODO probably need to add dpsi/dJ to the term below
        # W_dev_s = 2. * (grad_W_dev_nn[0] + 2. * grad_W_dev_nn[1]) * (J - 1.)
        W_dev_s = (
            2 * grad_W_dev_nn[0] + 4 * grad_W_dev_nn[1] + grad_W_dev_nn[2]
        ) * (J - 1)

        # return W_vol + (W_dev_nn - W_dev_0 - W_dev_s), state_old
        return W_dev_nn - W_dev_0 - W_dev_s, state_old

    def l0_regularization_gate(self, gate_key):
        gate_key, sub_key = jax.random.split(gate_key)
        params, static = eqx.partition(self.network, eqx.is_array)

        def theta_func(theta_bar, log_alpha):
            u = jax.random.uniform(gate_key, theta_bar.shape)
            # log_alpha = 0.01 * jax.random.normal(sub_key, theta_bar.shape)
            s = jax.nn.sigmoid(
                (jnp.log(u) - jnp.log(1 - u) + log_alpha) / self.beta
            )
            s_bar = (self.zeta - self.gamma) * s + self.gamma
            z = jnp.clip(s_bar, 0.0, 1.0)
            theta = theta_bar * z
            return theta

        params = jax.tree.map(
            lambda x, y: theta_func(x, y),
            params, self.log_alpha
        )
        params = eqx.combine(params, static)
        return params

    def l0_regularization_gate_test(self, gate_key):
        params, static = eqx.partition(self.network, eqx.is_array)

        def theta_func(theta_bar, log_alpha):
            # u = jax.random.uniform(gate_key, theta_bar.shape)
            # log_alpha = 0.01 * jax.random.normal(gate_key, theta_bar.shape)
            s = jax.nn.sigmoid(log_alpha) *\
                (self.zeta - self.gamma) + self.gamma
            z_hat = jnp.clip(s)
            return theta_bar * z_hat

        params = jax.tree.map(
            lambda x, y: theta_func(x, y),
            params, self.log_alpha
        )
        params = eqx.combine(params, static)
        return params

    def l0_regularization_term(self, gate_key, lambda_):
        params, _ = eqx.partition(self.network, eqx.is_array)

        def theta_func(theta_bar, log_alpha):
            # log_alpha = 0.01 * jax.random.normal(gate_key, theta_bar.shape)
            return jax.nn.sigmoid(
                log_alpha - self.beta * jnp.log(-self.gamma / self.zeta)
            )

        reg_terms = jax.tree.map(
            lambda x, y: theta_func(x, y),
            params, self.log_alpha
        )
        # return lambda_ * jnp.sum(reg_terms)
        leaves = jax.tree_util.tree_leaves(reg_terms)
        return lambda_ * sum(jnp.sum(g) for g in leaves)

    def num_effective_parameters(self, gate_key):
        theta = self.l0_regularization_gate_test(gate_key)
        params, static = eqx.partition(theta, eqx.is_array)
        leaves = jax.tree_util.tree_leaves(params)
        return sum(jnp.sum(g) for g in leaves)

    def parameter_enforcement(self):
        self = eqx.tree_at(
            lambda x: x.network,
            self,
            self.network.parameter_enforcement()
        )
        return self
