from jaxtyping import Array, Float
import equinox as eqx


Scalar = float
State = Float[Array, "ns"]
Tensor = Float[Array, "3 3"]
Vector = Float[Array, "3"]


class ConstitutiveModel(eqx.Module):
    def __repr__(self):
        prop_names = self.properties()
        string = f"{type(self)}:\n"
        max_str_length = max(map(len, prop_names))

        for prop_name in prop_names:
            v = getattr(self, prop_name)
            string = string + f"  {prop_name}"
            string = string.ljust(max_str_length)
            string = string + " = "
            if type(v) is float:
                string = string + f"{v}\n"
            elif issubclass(ConstitutiveModel, type(v)):
                string = string + f"{v.__repr__()}\n"
            else:
                string = string + f"{v.__repr__()}\n"

        return string

    def properties(self):
        return self.__dataclass_fields__

    def num_state_variables(self):
        return 0
