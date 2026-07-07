using LinearAlgebra
using MeshFreeExteriorCalculus
using SparseArrays

function make_boundary_mask(x, tol = 1e-12)
    xs = x[:,1]
    ys = x[:,2]
    return (abs.(xs .- 0.0) .< tol) .|
           (abs.(xs .- 1.0) .< tol) .|
           (abs.(ys .- 0.0) .< tol) .|
           (abs.(ys .- 1.0) .< tol)
end

# ------------------------------------------------------------
# Build point cloud / graph / MEEC complex
# ------------------------------------------------------------
grid = meshgrid(21, 21)
x = hcat(vec(grid.x), vec(grid.y))

hx = 1.0 / (21 - 1)
eps = 2.5 * hx

graph = EpsilonBallGraph(x, eps)
boundary_mask = make_boundary_mask(x)
complex = MEECComplex(graph, boundary_mask, 1.0)

# ------------------------------------------------------------
# Build gradient reconstruction
# ------------------------------------------------------------
gradrec = build_gradient_reconstruction(graph, complex.interior_mask)

# Lamé parameters
λ = 1.0
μ = 1.0

problem = LinearElasticityProblem(complex, gradrec, λ, μ)
ws = build_linear_elasticity_workspace(problem)

# ------------------------------------------------------------
# Manufactured affine solution
# ------------------------------------------------------------
A = [0.10  0.05;
    -0.02  0.08]

b = [0.0, 0.0]

g = affine_displacement_field(x, A, b)
f = affine_zero_body_force(x)

# exact U vector
N = size(x,1)
U_exact = zeros(eltype(x), 2N)
for i in 1:N
    U_exact[2i-1] = g[i,1]
    U_exact[2i]   = g[i,2]
end

# ------------------------------------------------------------
# Solve
# ------------------------------------------------------------
U = solve_linear_elasticity(problem, f, g; workspace=ws)

relerr = norm(U - U_exact) / norm(U_exact)
println("relative displacement error = ", relerr)

# ------------------------------------------------------------
# Residual of exact solution
# ------------------------------------------------------------
r_exact = apply_linear_elasticity_operator!(ws, U_exact, problem, f, g)
println("residual norm of exact affine solution = ", norm(r_exact))
