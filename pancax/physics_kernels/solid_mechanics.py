from abc import abstractmethod
from .base import BaseEnergyFormPhysics, element_pp, _output_names
from pancax.math import scalar_root_find
from pancax.math.tensor_math import tensor_2D_to_3D
import equinox as eqx
import jax.numpy as jnp


# different formulations e.g. plane strain/stress, axisymmetric etc.
class AbstractMechanicsFormulation(eqx.Module):
    n_dimensions: int = eqx.field(static=True)  # does this need to be static?

    @abstractmethod
    def modify_field_gradient(
        self, constitutive_model, grad_u, theta, state_old, dt
    ):
        pass


class PlaneStrain(AbstractMechanicsFormulation):
    n_dimensions: int = 2

    def extract_stress(self, P):
        return P[0:2, 0:2]

    def modify_field_gradient(
        self, constitutive_model, grad_u, theta, state_old, dt
    ):
        return tensor_2D_to_3D(grad_u)


class PlaneStress(AbstractMechanicsFormulation):
    n_dimensions: int
    settings: scalar_root_find.Settings

    def __init__(self):
        self.n_dimensions = 2
        self.settings = scalar_root_find.get_settings()

    def displacement_gradient(self, grad_u_33, grad_u):
        grad_u = jnp.array([
            [grad_u[0, 0], grad_u[0, 1], 0.],
            [grad_u[1, 0], grad_u[1, 1], 0.],
            [0., 0., grad_u_33]
        ])
        return grad_u

    def extract_stress(self, P):
        return P[0:2, 0:2]

    def modify_field_gradient(
        self, constitutive_model, grad_u, theta, state_old, dt
    ):
        def func(grad_u_33, constitutive_model, grad_u, theta, state_old, dt):
            grad_u = self.displacement_gradient(grad_u_33, grad_u)
            return constitutive_model.cauchy_stress(
                grad_u, theta, state_old, dt
            )[0][2, 2]

        def my_func(x):
            return func(x, constitutive_model, grad_u, theta, state_old, dt)

        # TODO make below options
        root_guess = 0.05
        root_bracket = jnp.array([-0.99, 10.])

        root, _ = scalar_root_find.find_root(
            my_func, root_guess, root_bracket, self.settings
        )
        grad_u = self.displacement_gradient(root, grad_u)
        return grad_u


class ThreeDimensional(AbstractMechanicsFormulation):
    n_dimensions: int = 3

    def extract_stress(self, P):
        return P

    def modify_field_gradient(
        self, constitutive_model, grad_u, theta, state_old, dt
    ):
        return grad_u


class SolidMechanics(BaseEnergyFormPhysics):
    field_value_names: tuple[str, ...]
    constitutive_model: any
    formulation: AbstractMechanicsFormulation

    def __init__(self, constitutive_model, formulation) -> None:
        # TODO clean this up below
        field_value_names = ("displ_x", "displ_y")
        super().__init__(field_value_names)
        if formulation.n_dimensions > 2:
            field_value_names = field_value_names + ("displ_z",)

        self.field_value_names = field_value_names
        self.constitutive_model = constitutive_model
        self.formulation = formulation

    def energy(self, params, x, t, u, grad_u, state_old, dt, *args):
        theta = 60.
        grad_u = self.formulation.modify_field_gradient(
            self.constitutive_model, grad_u, theta, state_old, dt
        )
        return self.constitutive_model.energy(grad_u, theta, state_old, dt)

    @property
    def num_state_variables(self):
        return self.constitutive_model.num_state_variables

    def update_var_name_to_method(self):
        new_pytree = super().update_var_name_to_method()

        var_name_to_method = new_pytree.var_name_to_method

        @eqx.filter_jit
        def _internal_force(p, d, t, u, s, dt, *args):
            return self.potential_energy_and_internal_force(
                p, d, t, u, s, dt, *args
            )[1]
        names = ("internal_force_x", "internal_force_y")
        if self.formulation.n_dimensions > 2:
            # names = (names, "internal_force_z")
            names = names + ("internal_force_z",)
        var_name_to_method["internal_force"] = {
            "method": _internal_force,
            "names": names
        }
        var_name_to_method["deformation_gradient"] = {
            "method": element_pp(
                self.constitutive_model.deformation_gradient,
                self,
                is_kinematic_method=True
            ),
            "names": _output_names("F", "full_tensor")
        }
        var_name_to_method["I1_bar"] = {
            "method": element_pp(
                self.constitutive_model.I1_bar,
                self,
                is_kinematic_method=True
            ),
            "names": _output_names("I1_bar", "scalar")
        }
        var_name_to_method["pk1_stress"] = {
            "method": element_pp(
                self.constitutive_model.pk1_stress,
                self,
                is_constitutive_method=True
            ),
            "names": _output_names("P", "full_tensor")
        }
        # special case handled just so parser doesn't break
        var_name_to_method["state_variables"] = {
            "method": element_pp(
                lambda x: x,
                self,
                is_state_method=True
            ),
            "names": tuple([
                f"state_variable_{n + 1}"
                for n in range(self.num_state_variables)
            ])
        }

        return new_pytree
