using MeshFreeExteriorCalculus

function make_boundary_mask(x, tol = 1e-12)
    xs = map(z -> z[1], eachrow(x))
    ys = map(z -> z[2], eachrow(x))
    return (abs.(xs .- 0.0) .< tol) .|
           (abs.(xs .- 1.0) .< tol) .|
           (abs.(ys .- 0.0) .< tol) .|
           (abs.(ys .- 1.0) .< tol)
end

grid = meshgrid(51, 51)
# x = hcat(vec(grid.x), vec(grid.y))' |> collect
x = hcat(vec(grid.x), vec(grid.y))
graph = EpsilonBallGraph(x, 0.05)
boundary_mask = make_boundary_mask(x)

complex = MEECComplex(graph, boundary_mask, 1.0)