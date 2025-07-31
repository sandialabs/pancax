from jaxtyping import Array, Float
import equinox as eqx


Scalar = float
State = Float[Array, "ns"]
Tensor = Float[Array, "3 3"]
Vector = Float[Array, "3"]


class ConstitutiveModel(eqx.Module):
    def __repr__(self):
        prop_names = self.properties()
        # props = asdict(self)
        # print(props)
        string = f"{type(self)}:\n"

        # for prop_name, prop in zip(prop_names, props):
        #     string = string + f"  {prop_name} = {prop}\n"
        # for k, v in props.items():
        max_str_length = max(map(len, prop_names))
        print(max_str_length)

        for prop_name in prop_names:
            v = getattr(self, prop_name)
            string = string + f"  {prop_name}"
            # string = string.rjust(max_str_length)
            # string = string + " = "
            string = string.ljust(max_str_length)
            string = string + " = "
            if type(v) is float:
                string = string + f"{v}\n"
            # elif type(v) is ConstitutiveModel:
            elif issubclass(ConstitutiveModel, type(v)):
                string = string + f"{v.__repr__()}\n"
            else:
                string = string + f"{v()}\n"
            # if type(v) is float:
            #     string = string + f"  {prop_name} = {v}\n"
            # else:
            #     string = string + f"  {prop_name} = {v()}\n"
        return string

    def properties(self):
        return self.__dataclass_fields__

    def num_state_variables(self):
        return 0
