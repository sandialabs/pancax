# from .bcs import *
from .bcs import DirichletBC, NeumannBC
from .bvps import SimpleShearLinearRamp, UniaxialTensionLinearRamp
from .constitutive_models import \
    ConstitutiveModel, \
    BoundedProperty, \
    FixedProperty, \
    Property, \
    BlatzKo, \
    Gent, \
    Hencky, \
    NeoHookean, \
    Swanson, \
    PronySeries, \
    SimpleFeFv, \
    WLF
from .data import FullFieldData, FullFieldDataLoader, GlobalData
from .domains import CollocationDomain, DeltaPINNDomain, VariationalDomain
from .fem import \
    DofManager, \
    FunctionSpace, \
    NonAllocatedFunctionSpace, \
    construct_function_space, \
    Mesh, \
    construct_mesh_from_basic_data, \
    create_edges, \
    create_higher_order_mesh_from_simplex_mesh, \
    create_nodesets_from_sidesets, \
    create_structured_mesh_data, \
    QuadratureRule, \
    read_exodus_mesh, \
    assemble_sparse_stiffness_matrix, \
    Hex8Element, \
    LineElement, \
    Quad4Element, \
    Quad9Element, \
    SimplexTriElement, \
    Tet4Element, \
    Tet10Element
from .history_writer import EnsembleHistoryWriter, HistoryWriter
from .logging import EnsembleLogger, Logger
from .loss_functions import \
    DirichletBCLoss, \
    NeumannBCLoss, \
    FullFieldDataLoss, \
    ICLossFunction, \
    StrongFormResidualLoss, \
    CombineLossFunctions, \
    EnergyLoss, \
    EnergyAndResidualLoss, \
    EnergyResidualAndReactionLoss, \
    ResidualMSELoss, \
    PathDependentEnergyLoss, \
    PathDependentEnergyResidualAndReactionLoss, \
    UserDefinedLossFunction
from .networks import \
    Field, \
    FieldPhysicsPair, \
    Linear, \
    MLP, \
    MLPBasis, \
    Network, \
    Parameters, \
    ResNet
from .optimizers import Adam, LBFGS
from .physics_kernels import \
    BasePhysics, \
    BaseEnergyFormPhysics, \
    BaseStrongFormPhysics, \
    BaseVariationalFormPhysics, \
    BeerLambertLaw, \
    BurgersEquation, \
    HeatEquation, \
    LaplaceBeltrami, \
    Poisson, \
    SolidMechanics, \
    BaseMechanicsFormulation, \
    IncompressiblePlaneStress, \
    PlaneStrain, \
    ThreeDimensional
from .post_processor import PostProcessor
from .problems import ForwardProblem, InverseProblem
# from .trainer import Trainer
from .utils import find_data_file, find_mesh_file
from jax import numpy as jnp
from jax import random
from pathlib import Path

import equinox as eqx
import jax
import matplotlib.pyplot as plt
import optax
import os

__all__ = [
    # bcs
    "DirichletBC",
    "NeumannBC",
    # bvps
    "SimpleShearLinearRamp",
    "UniaxialTensionLinearRamp",
    # constitutive models
    "ConstitutiveModel",
    "BoundedProperty",
    "FixedProperty",
    "Property",
    "BlatzKo",
    "Gent",
    "Hencky",
    "NeoHookean",
    "Swanson",
    "PronySeries",
    "SimpleFeFv",
    "WLF",
    # "data",
    "FullFieldData",
    "FullFieldDataLoader",
    "GlobalData",
    # domains
    "CollocationDomain",
    "DeltaPINNDomain",
    "VariationalDomain",
    # fem
    "DofManager",
    "FunctionSpace",
    "NonAllocatedFunctionSpace",
    "construct_function_space",
    "Mesh",
    "construct_mesh_from_basic_data",
    "create_edges",
    "create_higher_order_mesh_from_simplex_mesh",
    "create_nodesets_from_sidesets",
    "create_structured_mesh_data",
    "QuadratureRule",
    "read_exodus_mesh",
    "assemble_sparse_stiffness_matrix",
    # elements module
    "Hex8Element",
    "LineElement",
    "Quad4Element",
    "Quad9Element",
    "SimplexTriElement",
    "Tet4Element",
    "Tet10Element",
    # history writers
    "EnsembleHistoryWriter",
    "HistoryWriter",
    # loggers
    "EnsembleLogger",
    "Logger",
    # loss functions
    "DirichletBCLoss",
    "NeumannBCLoss",
    "FullFieldDataLoss",
    "ICLossFunction",
    "StrongFormResidualLoss",
    "CombineLossFunctions",
    "EnergyLoss",
    "EnergyAndResidualLoss",
    "EnergyResidualAndReactionLoss",
    "ResidualMSELoss",
    "PathDependentEnergyLoss",
    "PathDependentEnergyResidualAndReactionLoss",
    "UserDefinedLossFunction",
    # networks
    "Field",
    "FieldPhysicsPair",
    "Linear",
    "MLP",
    "MLPBasis",
    "Network",
    "Parameters",
    "ResNet",
    # optimizers
    "Adam",
    "LBFGS",
    # physics
    "BasePhysics",
    "BaseEnergyFormPhysics",
    "BaseStrongFormPhysics",
    "BaseVariationalFormPhysics",
    "BeerLambertLaw",
    "BurgersEquation",
    "HeatEquation",
    "LaplaceBeltrami",
    "Poisson",
    "BaseMechanicsFormulation",
    "IncompressiblePlaneStress",
    "PlaneStrain",
    "ThreeDimensional",
    "SolidMechanics",
    # post-processors
    "PostProcessor",
    # problems
    "ForwardProblem",
    "InverseProblem",
    # pancax helper methods
    "find_data_file",
    "find_mesh_file",
    # other helper modules
    "eqx",
    "jax",
    "jnp",
    "optax",
    "os",
    "plt",
    "random",
    "Path"
]
