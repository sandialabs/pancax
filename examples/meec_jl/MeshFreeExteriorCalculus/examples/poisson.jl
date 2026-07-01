using LinearAlgebra
using MeshFreeExteriorCalculus

function make_boundary_mask(x, tol = 1e-12)
    xs = map(z -> z[1], eachcol(x))
    ys = map(z -> z[2], eachcol(x))
    return (abs.(xs .- 0.0) .< tol) .|
           (abs.(xs .- 1.0) .< tol) .|
           (abs.(ys .- 0.0) .< tol) .|
           (abs.(ys .- 1.0) .< tol)
end

grid = meshgrid(101, 101)
x = hcat(vec(grid.x), vec(grid.y))' |> collect
graph = EpsilonBallGraph(x, 0.025)
boundary_mask = make_boundary_mask(x)
complex = MEECComplex(graph, boundary_mask, 1.0)

# u_exact = @. sin(pi * x[:,1]) * sin(pi * x[:,2])
u_exact = @. sin(pi * x[1, :]) * sin(pi * x[2, :])
f = @. 2 * pi^2 * u_exact
g = copy(u_exact)

model = ZeroFluxModel()
ws = build_nonlinear_workspace(complex)

u = solve_dirichlet_nonlinear(
    model,
    complex,
    f,
    g;
    ε=1.0,
    verbose=true,
    workspace=ws,
)

println("relative error = ", norm(u - u_exact) / norm(u_exact))
u