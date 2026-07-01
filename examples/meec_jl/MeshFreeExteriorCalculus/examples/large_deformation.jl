# using Exodus
# using LinearAlgebra
# using MeshFreeExteriorCalculus

# nx = 21
# ny = 21
# grid = meshgrid(nx, ny)

# # column-major points: (2,N)

# # x = vcat(reshape(grid.x, 1, :), reshape(grid.y, 1, :))
# exo = ExodusDatabase("unit_square_two_side_holes.e", "r")
# x = read_coordinates(exo)
# nsets = read_set.((exo,), (NodeSet,), ["bottom", "top"])
# @show nsets
# nodes = mapreduce(nset -> nset.nodes, vcat, nsets) |> unique |> sort
# display(nodes)
# @assert false

# h = 1.0 / (nx - 1)
# eps = 2.5 * h

# graph = EpsilonBallGraph(x, eps)
# boundary_mask = make_boundary_mask(x)
# complex = MEECComplex(graph, boundary_mask, 1.0)

# gradrec = build_gradient_reconstruction(graph, complex.interior_mask)

# λ = 1.0
# μ = 1.0

# problem = NeoHookeanProblem(complex, gradrec, λ, μ)
# ws = build_neo_hookean_workspace(problem)

# F0 = [1.10  0.10;
#       0.00  1.00]

# c = [0.0, 0.0]

# g = homogeneous_deformation_displacement(x, F0, c)   # (2,N)
# f = homogeneous_zero_body_force(x)                   # (2,N)

# N = num_nodes(graph)
# U_exact = vec(g)   # column-major flattening

# r_exact = apply_neo_hookean_operator!(ws, U_exact, problem, f, g)
# println("residual norm of exact homogeneous deformation = ", norm(r_exact))

# U = solve_neo_hookean_continuation(
#     problem,
#     f,
#     g;
#     nsteps = 10,
#     u0 = nothing,
#     max_newton = 12,
#     tol = 1e-8,
#     line_search_steps = 8,
#     verbose = true,
#     workspace = ws,
# )

# relerr = norm(U - U_exact) / norm(U_exact)
# println("relative displacement error = ", relerr)
###############################
# using Exodus
# using LinearAlgebra
# using MeshFreeExteriorCalculus

# # ============================================================
# # User inputs
# # ============================================================

# mesh_file = "unit_square_two_side_holes.e"

# # material
# λ = 1.0
# μ = 1.0

# # loading
# δ = 0.02

# # continuation / Newton
# nsteps = 20
# max_newton = 20
# tol = 1e-8
# line_search_steps = 10
# verbose = true

# # graph radius scaling
# eps_factor = 6.0


# # ============================================================
# # Helpers
# # ============================================================

# """
# Build a geometric boundary mask from named Exodus node sets.

# This mask is about geometric/domain boundary classification,
# not Dirichlet boundary conditions.
# """
# function build_geometric_boundary_mask(exo, x; names)
#     N = size(x, 2)
#     all_nodes = Int[]
#     for name in names
#         nset = read_set(exo, NodeSet, name)
#         append!(all_nodes, nset.nodes)
#     end
#     all_nodes = unique(sort(all_nodes))
#     mask = falses(N)
#     mask[all_nodes] .= true
#     return mask, all_nodes
# end


# """
# Flatten a nodal vector field Umat::(2,N) into column-major vector form.
# """
# flatten_field(Umat) = vec(Umat)


# """
# Reshape a flattened displacement vector U::(2N,) into (2,N).
# """
# unflatten_field(U) = reshape(U, 2, length(U) ÷ 2)


# """
# Construct a simple vertical interpolation initial guess:
#     ux = 0
#     uy = δ * (y - ymin)/(ymax - ymin)

# Then enforce Dirichlet BC values exactly.
# """
# function build_initial_guess(x, bc, δ)
#     N = size(x, 2)
#     U0 = zeros(eltype(x), 2N)
#     U0m = reshape(U0, 2, N)

#     ymin = minimum(x[2, :])
#     ymax = maximum(x[2, :])
#     Ly = ymax - ymin

#     @inbounds for i in 1:N
#         Y = x[2, i]
#         U0m[1, i] = 0.0
#         U0m[2, i] = δ * (Y - ymin) / Ly
#     end

#     # Enforce prescribed Dirichlet values exactly
#     @inbounds for i in 1:N
#         if bc.is_constrained[1, i]
#             U0m[1, i] = bc.values[1, i]
#         end
#         if bc.is_constrained[2, i]
#             U0m[2, i] = bc.values[2, i]
#         end
#     end

#     return U0
# end


# # ============================================================
# # Load mesh and classify boundaries
# # ============================================================

# exo = ExodusDatabase(mesh_file, "r")
# x = read_coordinates(exo)   # expected shape: (2, N)
# N = size(x, 2)

# # Node sets used for mechanics boundary conditions
# bottom_set = read_set(exo, NodeSet, "bottom")
# top_set    = read_set(exo, NodeSet, "top")

# bottom_nodes = sort(unique(bottom_set.nodes))
# top_nodes    = sort(unique(top_set.nodes))

# println("bottom nodes: ", length(bottom_nodes))
# println("top nodes:    ", length(top_nodes))

# # ------------------------------------------------------------
# # Geometric boundary classification
# # ------------------------------------------------------------
# # IMPORTANT:
# # This is about the domain boundary ∂Ω, not about Dirichlet BCs.
# # Hole boundaries are geometric boundaries too, even if traction-free.
# # ------------------------------------------------------------
# geometric_boundary_names = [
#     "bottom",
#     "top",
#     "left",
#     "right",
#     "left_hole",
#     "right_hole",
# ]

# geometric_boundary_mask, geometric_boundary_nodes =
#     build_geometric_boundary_mask(exo, x; names=geometric_boundary_names)

# println("geometric boundary nodes: ", length(geometric_boundary_nodes))

# close(exo)


# # ============================================================
# # Build graph / MEEC geometry / reconstruction
# # ============================================================

# xmin, xmax = minimum(x[1, :]), maximum(x[1, :])
# ymin, ymax = minimum(x[2, :]), maximum(x[2, :])
# area = (xmax - xmin) * (ymax - ymin)

# # crude characteristic spacing from point count
# h = sqrt(area / N)
# eps = eps_factor * h

# println("N = ", N)
# println("h ≈ ", h)
# println("eps = ", eps)

# graph = EpsilonBallGraph(x, eps)

# # IMPORTANT:
# # geometric_boundary_mask is used for domain-boundary classification
# # in the point-cloud/MEEC construction.
# complex = MEECComplex(graph, geometric_boundary_mask, area)

# # reconstruction used by continuum-faithful mechanics operator
# gradrec = build_gradient_reconstruction(graph, complex.interior_mask)

# # nonlinear mechanics problem
# problem = NeoHookeanProblem(complex, gradrec, λ, μ)
# ws = build_neo_hookean_workspace(problem)


# # ============================================================
# # Body force
# # ============================================================

# # For this test, no body force
# f = zeros(eltype(x), 2, N)


# # ============================================================
# # Dirichlet boundary conditions (mechanics)
# # ============================================================
# # IMPORTANT:
# # This is separate from geometric_boundary_mask.
# #
# # Here we choose a mechanically standard displacement-controlled test:
# #
# #   bottom: all uy = 0
# #   bottom: one node ux = 0   (remove rigid translation)
# #   top:    all uy = δ
# #
# # top ux is left free, so the body can contract/expand laterally.
# # ============================================================

# bc = build_dirichlet_bc_data(eltype(x), 2, N)

# # bottom: uy = 0
# constrain_dofs!(bc, bottom_nodes, [2], 0.0)

# # one anchor point in ux on the bottom to remove rigid horizontal translation
# anchor_node = bottom_nodes[argmin(x[1, bottom_nodes])]
# constrain_dofs!(bc, [anchor_node], [1], 0.0)

# # top: uy = δ
# constrain_dofs!(bc, top_nodes, [1], 0.0)
# constrain_dofs!(bc, top_nodes, [2], δ)

# println("anchor node for ux = ", anchor_node)


# # ============================================================
# # Initial guess
# # ============================================================

# U0 = build_initial_guess(x, bc, δ)


# # ============================================================
# # Solve with continuation
# # ============================================================

# U = solve_neo_hookean_continuation(
#     problem,
#     f,
#     bc;
#     nsteps = nsteps,
#     u0 = U0,
#     max_newton = max_newton,
#     tol = tol,
#     line_search_steps = line_search_steps,
#     verbose = verbose,
#     workspace = ws,
# )

# println("solve complete")


# # ============================================================
# # Postprocess
# # ============================================================

# Um = unflatten_field(U)
# ux = Um[1, :]
# uy = Um[2, :]

# println("max |ux| = ", maximum(abs.(ux)))
# println("max |uy| = ", maximum(abs.(uy)))
# println("min ux   = ", minimum(ux))
# println("max ux   = ", maximum(ux))
# println("min uy   = ", minimum(uy))
# println("max uy   = ", maximum(uy))


# # ============================================================
# # Write Exodus output
# # ============================================================

# copy_mesh(mesh_file, "output.exo")
# exo_out = ExodusDatabase("output.exo", "rw")

# write_time(exo_out, 1, 0.0)
# write_names(exo_out, NodalVariable, ["displ_x", "displ_y"])
# write_values(exo_out, NodalVariable, 1, "displ_x", ux)
# write_values(exo_out, NodalVariable, 1, "displ_y", uy)

# close(exo_out)

# println("Wrote displacement output to output.exo")


using Exodus
using LinearAlgebra
using MeshFreeExteriorCalculus

# ============================================================
# User inputs
# ============================================================

mesh_file = "unit_square_two_side_holes.e"

# material
λ = 1.0
μ = 1.0

# displacement-controlled loading
δ = 0.05

# continuation / Newton
nsteps = 100
max_newton = 15
tol = 1e-8
line_search_steps = 12
verbose = true

# graph radius factor
eps_factor = 2.5
# output
output_file = "output_energy_analytic.exo"


# ============================================================
# Helpers
# ============================================================

"""
Build geometric boundary mask from named Exodus node sets.

This is NOT the same as the Dirichlet BC set.
It marks nodes that lie on the geometric/domain boundary.
"""
function build_geometric_boundary_mask(exo, x; names)
    N = size(x, 2)
    all_nodes = Int[]
    for name in names
        nset = read_set(exo, NodeSet, name)
        append!(all_nodes, nset.nodes)
    end
    all_nodes = unique(sort(all_nodes))
    mask = falses(N)
    mask[all_nodes] .= true
    return mask, all_nodes
end


"""
Build a simple vertical interpolation initial guess:
    ux = 0
    uy = δ * (y - ymin)/(ymax - ymin)

Then enforce the Dirichlet values exactly.
"""
function build_initial_guess(x, bc, δ)
    N = size(x, 2)
    U0 = zeros(eltype(x), 2N)
    U0m = reshape(U0, 2, N)

    ymin = minimum(x[2,:])
    ymax = maximum(x[2,:])
    Ly = ymax - ymin

    @inbounds for i in 1:N
        Y = x[2,i]
        U0m[1,i] = 0.0
        U0m[2,i] = δ * (Y - ymin) / Ly
    end

    enforce_dirichlet_values!(U0, bc)
    return U0
end


# ============================================================
# Load Exodus mesh
# ============================================================

exo = ExodusDatabase(mesh_file, "r")
x = read_coordinates(exo)   # expected shape: (2, N)
N = size(x, 2)

# Node sets for BCs
bottom_set = read_set(exo, NodeSet, "bottom")
top_set    = read_set(exo, NodeSet, "top")

bottom_nodes = sort(unique(bottom_set.nodes))
top_nodes    = sort(unique(top_set.nodes))

println("bottom nodes: ", length(bottom_nodes))
println("top nodes:    ", length(top_nodes))

# ------------------------------------------------------------
# Geometric boundary classification
# ------------------------------------------------------------
# These are all nodes on the actual domain boundary ∂Ω,
# including the hole boundaries.
# ------------------------------------------------------------
geometric_boundary_names = [
    "bottom",
    "top",
    "left",
    "right",
    "left_hole",
    "right_hole",
]

geometric_boundary_mask, geometric_boundary_nodes =
    build_geometric_boundary_mask(exo, x; names=geometric_boundary_names)

println("geometric boundary nodes: ", length(geometric_boundary_nodes))

close(exo)


# ============================================================
# Graph / reconstruction / nodal quadrature
# ============================================================

xmin, xmax = minimum(x[1,:]), maximum(x[1,:])
ymin, ymax = minimum(x[2,:]), maximum(x[2,:])
# area = (xmax - xmin) * (ymax - ymin)
# area = 0.9293142
area = 1.0 - 2π * (0.15)^2

# crude characteristic spacing
h = sqrt(area / N)
eps = eps_factor * h

println("N     = ", N)
println("area  = ", area)
println("h     ≈ ", h)
println("eps   = ", eps)

graph = EpsilonBallGraph(x, eps)

# Reconstruction uses the interior mask, so geometric boundary classification matters
interior_mask = .!geometric_boundary_mask
gradrec = build_gradient_reconstruction(graph, interior_mask)

# Nodal quadrature weights
m0 = build_virtual_volumes(graph, area, interior_mask)
# m0 = fill(eltype(x)(area / N), N)

println("sum(m0) = ", sum(m0))


# ============================================================
# Hyperelastic problem
# ============================================================

problem = NeoHookeanEnergyProblem(graph, gradrec, m0, λ, μ)
ws = build_neo_hookean_energy_workspace(problem)

# body force
f = zeros(eltype(x), 2, N)


# ============================================================
# Dirichlet BCs
# ============================================================
# Mechanics BCs:
#   bottom: uy = 0
#   one bottom node: ux = 0
#   top: uy = δ
#
# This allows lateral motion (ux free except for the anchor),
# which is much more appropriate than fixing ux on all top/bottom nodes.
# ============================================================

bc = build_dirichlet_bc_data(eltype(x), 2, N)

# bottom: uy = 0
constrain_dofs!(bc, bottom_nodes, [1, 2], 0.0)

# one bottom node: ux = 0 (anchor to remove rigid horizontal motion)
# anchor_node = bottom_nodes[argmin(x[1, bottom_nodes])]
# constrain_dofs!(bc, [anchor_node], [1], 0.0)

# top: uy = δ
constrain_dofs!(bc, top_nodes, [1], 0.0)
constrain_dofs!(bc, top_nodes, [2], δ)

# println("anchor node for ux = ", anchor_node)


# ============================================================
# Initial guess
# ============================================================

U0 = build_initial_guess(x, bc, δ)


# ============================================================
# Solve by continuation
# ============================================================

U = solve_neo_hookean_energy_continuation(
    problem,
    f,
    bc;
    nsteps = nsteps,
    u0 = nothing,
    max_newton = max_newton,
    tol = tol,
    line_search_steps = line_search_steps,
    verbose = verbose,
    workspace = ws,
)

println("solve complete")


# ============================================================
# Postprocess
# ============================================================

Um = reshape(U, 2, N)
ux = Um[1,:]
uy = Um[2,:]

println("max |ux| = ", maximum(abs.(ux)))
println("max |uy| = ", maximum(abs.(uy)))
println("min ux   = ", minimum(ux))
println("max ux   = ", maximum(ux))
println("min uy   = ", minimum(uy))
println("max uy   = ", maximum(uy))

# sanity checks on prescribed boundaries
println("bottom uy max abs = ", maximum(abs.(uy[bottom_nodes])))
println("top uy min/max    = ", minimum(uy[top_nodes]), ", ", maximum(uy[top_nodes]))


# ============================================================
# Write Exodus output
# ============================================================

copy_mesh(mesh_file, output_file)
exo_out = ExodusDatabase(output_file, "rw")

write_time(exo_out, 1, 0.0)
write_names(exo_out, NodalVariable, ["displ_x", "displ_y"])

write_values(exo_out, NodalVariable, 1, "displ_x", ux)
write_values(exo_out, NodalVariable, 1, "displ_y", uy)

close(exo_out)

println("Wrote displacement output to ", output_file)