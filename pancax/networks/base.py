import equinox as eqx


class BasePancaxModel(eqx.Module):
    """
    Base class for pancax model parameters.

    This includes a few helper methods
    """

    def serialise(self, base_name, epoch):
        file_name = f"{base_name}_{str(epoch).zfill(7)}.eqx"
        print(f"Serialising current parameters to {file_name}")
        eqx.tree_serialise_leaves(file_name, self)
