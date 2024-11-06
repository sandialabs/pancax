from .function_space import FunctionSpace
from jaxtyping import Array, Bool, Float, Int
from pancax.bcs import EssentialBC
from pancax.timer import Timer
from typing import List, Tuple
import jax.numpy as np
import numpy as onp

# TODO
# getting some error when making this a child of eqx.Module
class DofManager:
    """
    Collection of arrays needed to differentiate between
    fixed and free dofs for fem like calculations.

    TODO better document the parameters in this guy
    """
    def __init__(self, mesh, dim: int, EssentialBCs: List[EssentialBC]) -> None:
        """
        :param functionSpace: ``FunctionSpace`` object
        :param dim: The number of dims (really the number of active dofs for the physics)
        :param EssentialBCs: A list of of ``EssentialBC`` objects
        """
        with Timer('DofManager.__init__'):
            self.fieldShape = mesh.num_nodes, dim
            self.isBc = onp.full(self.fieldShape, False, dtype=bool)
            for ebc in EssentialBCs:
                self.isBc[mesh.nodeSets[ebc.nodeSet], ebc.component] = True

            self.isUnknown = ~self.isBc

            self.ids = onp.arange(self.isBc.size).reshape(self.fieldShape)

            self.unknownIndices = self.ids[self.isUnknown]
            self.bcIndices = self.ids[self.isBc]

            ones = onp.ones(self.isBc.size, dtype=int) * -1
            # self.dofToUnknown = ones.at[self.unknownIndices].set(np.arange(self.unknownIndices.size)) 
            self.dofToUnknown = ones
            self.dofToUnknown[self.unknownIndices] = onp.arange(self.unknownIndices.size)

            self.HessRowCoords, self.HessColCoords = self._make_hessian_coordinates(onp.array(mesh.conns))

            self.hessian_bc_mask = self._make_hessian_bc_mask(onp.array(mesh.conns))

    def get_bc_size(self) -> int:
        """
        :return: the number of fixed dofs
        """
        return np.sum(self.isBc).item() # item() method casts to Python int

    def get_unknown_size(self) -> int:
        """
        :return: the size of the unkowns vector
        """
        return np.sum(self.isUnknown).item() # item() method casts to Python int

    def create_field(self, Uu, Ubc=0.0) -> Float[Array, "nn nd"]:
        """
        :param Uu: Vector of unknown values
        :param Ubc: Values for bc to apply
        :return: U, a field of unknowns and bcs combined.
        """
        U = np.zeros(self.isBc.shape).at[self.isBc].set(Ubc)
        return U.at[self.isUnknown].set(Uu)

    def get_bc_values(self, U) -> Float[Array, "nb"]:
        """
        :param U: a nodal field
        :return: the bc values in the field U
        """
        return U[self.isBc]

    def get_unknown_values(self, U) -> Float[Array, "nu"]:
        """
        :param U: a nodal field
        :return: the unknown values in the field U
        """
        return U[self.isUnknown]

    def slice_unknowns_with_dof_indices(
        self, Uu: Float[Array, "nu"], dofIndexSlice: Int[Array, "nn"]
    ) -> Float[Array, "nu_new"]:
        i = self.isUnknown[dofIndexSlice]
        j = self.dofToUnknown.reshape(self.fieldShape)[dofIndexSlice]
        return Uu[j[i]]

    def _make_hessian_coordinates(self, conns: Int[Array, "ne nnpe"]) -> Tuple[Int[Array, "nn"], Int[Array, "nn"]]:
        nElUnknowns = onp.zeros(conns.shape[0], dtype=int)
        nHessianEntries = 0
        for e, eNodes in enumerate(conns):
            elUnknownFlags = self.isUnknown[eNodes,:].ravel()
            nElUnknowns[e] = onp.sum(elUnknownFlags)
            nHessianEntries += onp.square(nElUnknowns[e])

        rowCoords = onp.zeros(nHessianEntries, dtype=int)
        colCoords = rowCoords.copy()
        rangeBegin = 0
        for e,eNodes in enumerate(conns):
            elDofs = self.ids[eNodes,:]
            elUnknownFlags = self.isUnknown[eNodes,:]
            elUnknowns = self.dofToUnknown[elDofs[elUnknownFlags]]
            elHessCoords = onp.tile(elUnknowns, (nElUnknowns[e],1))

            rangeEnd = rangeBegin + onp.square(nElUnknowns[e])

            rowCoords[rangeBegin:rangeEnd] = elHessCoords.ravel()
            colCoords[rangeBegin:rangeEnd] = elHessCoords.T.ravel()

            rangeBegin += onp.square(nElUnknowns[e])
        return rowCoords, colCoords

    def _make_hessian_bc_mask(self, conns: Int[Array, "ne nnpe"]) -> Bool[Array, "ne ndpe ndpe"]:
        nElements, nNodesPerElement = conns.shape
        nFields = self.ids.shape[1]
        nDofPerElement = nNodesPerElement*nFields

        hessian_bc_mask = onp.full((nElements,nDofPerElement,nDofPerElement),
                                   True, dtype=bool)
        for e, eNodes in enumerate(conns):
            eFlag = self.isBc[eNodes,:].ravel()
            hessian_bc_mask[e,eFlag,:] = False
            hessian_bc_mask[e,:,eFlag] = False
        return hessian_bc_mask
