from abc import abstractmethod
from .base import BaseEnergyFormPhysics, element_pp
from pancax.math.tensor_math import tensor_2D_to_3D
import equinox as eqx
import jax.numpy as jnp


# different formulations e.g. plane strain/stress, axisymmetric etc.
class BaseMechanicsFormulation(eqx.Module):
    n_dimensions: int = eqx.field(static=True)  # does this need to be static?

    @abstractmethod
    def modify_field_gradient(self, grad_u):
        pass


# note for this formulation we're getting NaNs if the
# reference configuration is used during calculation
# of the loss function
class IncompressiblePlaneStress(BaseMechanicsFormulation):
    n_dimensions = 2

    def __init__(self) -> None:
        print(
            "WARNING: Do not include a time of 0.0 with this formulation. "
            "You will get NaNs."
        )

    def deformation_gradient(self, grad_u):
        F = tensor_2D_to_3D(grad_u) + jnp.eye(3)
        F = F.at[2, 2].set(1.0 / jnp.linalg.det(grad_u + jnp.eye(2)))
        return F

    def modify_field_gradient(self, grad_u):
        F = self.deformation_gradient(grad_u)
        return F - jnp.eye(3)


class PlaneStrain(BaseMechanicsFormulation):
    n_dimensions: int = 2

    def extract_stress(self, P):
        return P[0:2, 0:2]

    def modify_field_gradient(self, grad_u):
        return tensor_2D_to_3D(grad_u)


class ThreeDimensional(BaseMechanicsFormulation):
    n_dimensions: int = 3

    def modify_field_gradient(self, grad_u):
        return grad_u


class SolidMechanics(BaseEnergyFormPhysics):
    field_value_names: tuple[str, ...]
    constitutive_model: any
    formulation: BaseMechanicsFormulation

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
        grad_u = self.formulation.modify_field_gradient(grad_u)
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
            names = (names, "internal_force_z")
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
            "names": (
                "F_xx", "F_xy", "F_xz",
                "F_yx", "F_yy", "F_yz",
                "F_zx", "F_zy", "F_zz"
            )
        }
        var_name_to_method["I1_bar"] = {
            "method": element_pp(
                self.constitutive_model.I1_bar,
                self,
                is_kinematic_method=True
            ),
            "names": ("I1_bar",)
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
