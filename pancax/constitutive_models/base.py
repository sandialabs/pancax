from jaxtyping import Array, Float
import equinox as eqx


Scalar = float
State = Float[Array, "ns"]
Tensor = Float[Array, "3 3"]
Vector = Float[Array, "3"]


class ConstitutiveModel(eqx.Module):
    def properties(self):
        return self.__dataclass_fields__

    def num_state_variables(self):
        return 0
