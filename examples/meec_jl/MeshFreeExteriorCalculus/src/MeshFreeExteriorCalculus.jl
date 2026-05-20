module MeshFreeExteriorCalculus

export EpsilonBallGraph
export MEECComplex
export meshgrid

using LinearAlgebra
using SparseArrays

function meshgrid(
    nx::Int, ny::Int;
    xmin = 0.0, xmax = 1.0,
    ymin = 0.0, ymax = 1.0
)
    x = LinRange(xmin, xmax, nx)
    y = LinRange(ymin, ymax, ny)
    return meshgrid(x, y)
end

function meshgrid(xin::LinRange{T, I}, yin::LinRange{T, I}) where {T <: Number, I <: Integer}
    nx = length(xin)
    ny = length(yin)
    xout = zeros(ny, nx)
    yout = zeros(ny, nx)
    for jx = 1:nx
        for ix = 1:ny
            xout[ix, jx] = xin[jx]
            yout[ix, jx] = yin[ix]
        end
    end
    return (x = xout, y = yout)
end

struct EpsilonBallGraph{
    I  <: Integer,
    T  <: Number,
    IV <: AbstractVector{I},
    RV <: AbstractVector{T},
    RM <: AbstractMatrix{T}
}
    ϵ::T
    senders::IV     # (E,)
    receivers::IV   # (E,)
    x::RM           # (2, N)
    dx::RM          # (2, E)
    r::RV           # (E,)
    midpoints::RM   # (E, 2)
    tangents::RM    # (E, 2)
end

# function EpsilonBallGraph(x::Matrix{T}, ϵ::T) where T <: Number
#     @assert size(x, 1) == 2
#     N = size(x, 2)
#     diff = [
#         x[k, j] - x[k, i]
#         for i in axes(x,2),
#             j in axes(x,2),
#             k in axes(x,1)
#     ]
#     dist = sqrt.(sum(abs2, diff; dims=3))[:, :, 1]
#     ii = Int[]
#     jj = Int[]

#     for j in 2:N
#         for i in 1:j - 1
#             push!(ii, i)
#             push!(jj, j)
#         end
#     end

#     senders = Int[]
#     receivers = Int[]

#     for k in eachindex(ii)
#         i = ii[k]
#         j = jj[k]
#         if dist[i, j] < ϵ
#             push!(senders, i)
#             push!(receivers, j)
#         end
#     end
#     E = length(senders)
#     #
#     # Edge vectors
#     #
#     dx = Matrix{T}(undef, 2, E)
#     for e in 1:E
#         for j in axes(dx, 1)
#             dx[j, e] = x[j, receivers[e]] - x[j, senders[e]]
#         end
#     end
#     #
#     # Edge lengths
#     #
#     r = vec(sqrt.(sum(abs2, dx; dims=1)))
#     #
#     # Tangents
#     #
#     tangents = similar(dx)
#     for e in 1:E
#         for j in axes(tangents, 1)
#             tangents[j, e] = dx[j, e] / max(r[e], T(1e-12))
#         end
#     end
#     #
#     # Midpoints
#     #
#     midpoints = Matrix{T}(undef, 2, E)
#     for e in 1:E
#         midpoints[:, e] .= 0.5 .* (
#             x[:, senders[e]] +
#             x[:, receivers[e]]
#         )
#     end

#     return EpsilonBallGraph(
#         ϵ,
#         senders, receivers,
#         x, dx, r,
#         midpoints, tangents
#     )
# end
function EpsilonBallGraph(x::Matrix{T}, ϵ::T) where {T <: Number}

    #
    # PyTorch layout:
    #
    # x :: (N, 2)
    #
    @assert size(x, 2) == 2

    N = size(x, 1)

    #
    # diff[i,j,k] = x[j,k] - x[i,k]
    #
    diff = [
        x[j,k] - x[i,k]
        for i in axes(x,1),
            j in axes(x,1),
            k in axes(x,2)
    ]

    #
    # Pairwise distances
    #
    dist = dropdims(
        sqrt.(sum(abs2, diff; dims=3));
        dims=3
    )

    #
    # Upper triangular indices (i < j)
    #
    ii = Int[]
    jj = Int[]

    for j in 2:N
        for i in 1:j-1
            push!(ii, i)
            push!(jj, j)
        end
    end

    #
    # Epsilon filtering
    #
    senders = Int[]
    receivers = Int[]

    for k in eachindex(ii)

        i = ii[k]
        j = jj[k]

        if dist[i,j] < ϵ
            push!(senders, i)
            push!(receivers, j)
        end
    end

    E = length(senders)

    #
    # dx :: (E, 2)
    #
    dx = Matrix{T}(undef, E, 2)

    for e in 1:E
        dx[e,:] .= x[receivers[e],:] .- x[senders[e],:]
    end

    #
    # r :: (E,)
    #
    r = vec(
        dropdims(
            sqrt.(sum(abs2, dx; dims=2));
            dims=2
        )
    )

    #
    # tangents :: (E, 2)
    #
    tangents = similar(dx)

    for e in 1:E
        tangents[e,:] .= dx[e,:] ./ max(r[e], T(1e-12))
    end

    #
    # midpoints :: (E, 2)
    #
    midpoints = Matrix{T}(undef, E, 2)

    for e in 1:E
        midpoints[e,:] .= 0.5 .* (
            x[senders[e],:] +
            x[receivers[e],:]
        )
    end

    return EpsilonBallGraph(
        ϵ,
        senders,
        receivers,
        x,
        dx,
        r,
        midpoints,
        tangents
    )
end

num_edges(graph::EpsilonBallGraph) = length(graph.senders)
num_nodes(graph::EpsilonBallGraph) = size(graph.x, 1)

struct MEECComplex{
    I  <: Integer,
    T  <: Number,
    IV <: AbstractVector{I},
    RV <: AbstractVector{T},
    RM <: AbstractMatrix{T},
    Mask
}
    graph::EpsilonBallGraph{I, T, IV, RV, RM}
    boundary_mask::Mask
    interior_mask::Mask
    d0::RM # TODO make sparse matrix
    m0::RV
    m1::RV
end

function MEECComplex(graph, boundary_mask, domain_volume, ridge = 1e-10)
    interior_mask = .!boundary_mask
    d0 = _build_incidence_matrix(graph)
    m0 = _build_virtual_volumes(graph, domain_volume, interior_mask)
    m1 = _solve_edge_areas_moment_matching(graph, interior_mask, m0, ridge)
    return MEECComplex(graph, boundary_mask, interior_mask, d0, m0, m1)
end

function _build_incidence_matrix(graph::EpsilonBallGraph)
    E = num_edges(graph)
    N = num_nodes(graph)
    d0 = zeros(E, N)
    for e in 1:E
        d0[e, graph.senders[e]] = -1.0
        d0[e, graph.receivers[e]] = 1.0
    end
    return d0
end

function _build_virtual_volumes(
    graph::EpsilonBallGraph,
    domain_volume,
    interior_mask
)
    N = size(graph.x, 1)
    ϕ_e = _wendland_like_kernel.(graph.r, (graph.ϵ,))
    #
    # Accumulate edge weights onto nodes
    #
    κ = zeros(eltype(graph.r), N)
    for e in eachindex(graph.senders)

        s = graph.senders[e]
        r = graph.receivers[e]

        ϕ = ϕ_e[e]

        κ[s] += ϕ
        κ[r] += ϕ
    end
    #
    # Inverse kappa on interior only
    #
    inv_κ = zeros(eltype(graph.r), N)
    for i in eachindex(interior_mask)
        if interior_mask[i]
            inv_κ[i] = 1.0 / max(κ[i], 1e-12)
        end
    end
    #
    # Normalize
    #
    inv_sum = max(sum(inv_κ), 1e-12)
    #
    # Virtual volumes
    #
    m = zeros(eltype(graph.r), N)
    for i in eachindex(interior_mask)
        if interior_mask[i]
            m[i] = domain_volume * inv_κ[i] / inv_sum
        end
    end

    return m
end

function _incident_edges(graph::EpsilonBallGraph, node_idx::Int)

    inc_sender = (graph.senders .== node_idx)
    inc_receiver = (graph.receivers .== node_idx)

    inc = inc_sender .| inc_receiver

    edge_idx = findall(inc)

    return edge_idx, inc_sender, inc_receiver
end

function _solve_edge_areas_moment_matching(
    graph::EpsilonBallGraph,
    interior_mask,
    m0,
    ridge
)

    E = length(graph.senders)
    N = size(graph.x, 1)

    ϕ = [(max(1.0 - r / graph.ϵ, 0.0)^2 + 1e-12) for r in graph.r]

    #
    # Sparse triplets for B
    #
    rows = Int[]
    cols = Int[]
    vals = Float64[]
    rhs  = Float64[]

    row_counter = 1

    for i in 1:N
        if !interior_mask[i]
            continue
        end

        edge_idx, _, _ = _incident_edges(graph, i)
        isempty(edge_idx) && continue

        ne = length(edge_idx)

        #
        # eta: (ne, 2)
        #
        eta = zeros(Float64, ne, 2)

        for (k, e) in enumerate(edge_idx)
            if graph.senders[e] == i
                eta[k, :] .= graph.dx[e, :]
            else
                eta[k, :] .= -graph.dx[e, :]
            end
        end

        #
        # 5 constraint rows per node
        #
        local_rows = (
            eta[:, 1],
            eta[:, 2],
            eta[:, 1] .* eta[:, 1],
            eta[:, 1] .* eta[:, 2],
            eta[:, 2] .* eta[:, 2],
        )

        local_rhs = (
            0.0,
            0.0,
            2.0 * m0[i],
            0.0,
            2.0 * m0[i],
        )

        for (rr, b) in zip(local_rows, local_rhs)

            for (k, e) in enumerate(edge_idx)
                push!(rows, row_counter)
                push!(cols, e)
                push!(vals, rr[k])
            end

            push!(rhs, b)
            row_counter += 1
        end
    end

    C = row_counter - 1

    if C == 0
        return ones(Float64, E)
    end

    #
    # Build sparse matrices
    #
    B = sparse(rows, cols, vals, C, E)

    Φ = spdiagm(0 => ϕ)

    S = B * Φ * B' + ridge * I(C)

    #
    # Solve Schur system
    #
    λ = S \ rhs

    #
    # Recover edge weights
    #
    a = Φ * (B' * λ)

    return a
end

function _wendland_like_kernel(r::T, ϵ::T) where T <: Number
    z = one(r) - r / ϵ
    if z > zero(z)
        return z * z
    else
        return zero(z)
    end
end

end # module MeshFreeExteriorCalculus
