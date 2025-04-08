from abc import abstractmethod
from ..base import HyperelasticModel, MechanicsModel
from ...base import ConstitutiveModel, Scalar, State, Tensor
from jaxtyping import Array, Float
from ...properties import Property
import equinox as eqx
import jax.numpy as jnp


# has general bookkeeping methods
class PathDependentModel(ConstitutiveModel):
    def extract_scalar(self, state: State, start_index: int) -> Scalar:
        return state[start_index]

    def extract_tensor(self, state: State, start_index: int) -> Tensor:
        return state[start_index:start_index + 9].reshape((3, 3))

    @abstractmethod
    def initial_state(self) -> State:
        pass


class PronySeries(eqx.Module):
    moduli: Float[Array, "nprony"] = eqx.field(converter=jnp.asarray)
    relaxation_times: Float[Array, "nprony"] = eqx.field(converter=jnp.asarray)

    def __init__(self, moduli, relaxation_times) -> None:
        assert len(moduli) == len(relaxation_times)
        self.moduli = moduli
        self.relaxation_times = relaxation_times

    def __len__(self) -> int:
        return len(self.moduli)


class ShiftFactorModel(ConstitutiveModel):
    def __call__(
        self,
        grad_u: Tensor, theta: Scalar, state: State, dt: Scalar
    ):
        return self.shift_factor(grad_u, theta, state, dt)

    @abstractmethod
    def shift_factor(
        self, grad_u: Tensor, theta: Scalar, state: State, dt: Scalar
    ) -> Scalar:
        pass


class WLF(ShiftFactorModel):
    C1: Property
    C2: Property
    theta_ref: Property

    def shift_factor(
        self, grad_u: Tensor, theta: Scalar, state: State, dt: Scalar
    ) -> Scalar:
        C1, C2, theta_ref = self.C1, self.C2, self.theta_ref
        loga_T = -C1 * (theta - theta_ref) / (C2 + (theta - theta_ref))
        return 10**loga_T


class HyperViscoElastic(MechanicsModel, PathDependentModel):
    eq_model: HyperelasticModel
    prony_series: PronySeries
    shift_factor_model: ShiftFactorModel

    def num_prony_terms(self) -> int:
        return len(self.prony_series)
