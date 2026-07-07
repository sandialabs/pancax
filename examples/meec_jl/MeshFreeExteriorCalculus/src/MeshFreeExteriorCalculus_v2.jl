module MeshFreeExteriorCalculus

using LinearAlgebra
using SparseArrays

# ============================================================
# Exports
# ============================================================

export meshgrid
export make_boundary_mask

export EpsilonBallGraph
export MEECComplex
export num_nodes
export num_edges

export GradientReconstruction
export build_gradient_reconstruction
export apply_scalar_gradient!

export DirichletBCData
export build_dirichlet_bc_data
export constrain_dofs!

export NeoHookeanProblem
export NeoHookeanWorkspace
export build_neo_hookean_workspace
export apply_neo_hookean_operator!
export assemble_neo_hookean_jacobian_fd!
export solve_neo_hookean
export solve_neo_hookean_continuation

export affine_displacement_field
export affine_zero_body_force
export homogeneous_deformation_displacement
export homogeneous_zero_body_force

# ============================================================
# Utilities
# ============================================================

function meshgrid(
    nx::Int, ny::Int;
    xmin = 0.0, xmax = 1.0,
    ymin = 0.0, ymax = 1.0
)
    x = LinRange(xmin, xmax, nx)
    y = LinRange(ymin, ymax, ny)
    return meshgrid(x, y)
end

function meshgrid(xin::LinRange{T,I}, yin::LinRange{T,I}) where {T<:Number,I<:Integer}
    nx = length(xin)
    ny = length(yin)
    xout = zeros(T, ny, nx)
    yout = zeros(T, ny, nx)
    for j in 1:nx
        for i in 1:ny
            xout[i,j] = xin[j]
            yout[i,j] = yin[i]
        end
    end
    return (x=xout, y=yout)
end

function make_boundary_mask(x::AbstractMatrix{T}, tol = T(1e-12)) where {T<:Number}
    @assert size(x,1) == 2
    N = size(x,2)
    mask = falses(N)
    @inbounds for i in 1:N
        xi = x[1,i]
        yi = x[2,i]
        mask[i] =
            (abs(xi - zero(T)) < tol) ||
            (abs(xi - one(T))  < tol) ||
            (abs(yi - zero(T)) < tol) ||
            (abs(yi - one(T))  < tol)
    end
    return mask
end

@inline function _wendland_like_kernel(r::T, ϵ::T) where {T<:Number}
    z = one(T) - r / ϵ
    return z > zero(T) ? z*z : zero(T)
end

num_nodes(x::AbstractMatrix) = size(x,2)

# ============================================================
# Epsilon-ball graph (column-major geometry)
# ============================================================

struct EpsilonBallGraph{
    I  <: Integer,
    T  <: Number,
    IV <: AbstractVector{I},
    RV <: AbstractVector{T},
    RM <: AbstractMatrix{T}
}
    ϵ::T
    senders::IV      # (E,)
    receivers::IV    # (E,)
    x::RM            # (2,N)
    dx::RM           # (2,E)
    r::RV            # (E,)
    midpoints::RM    # (2,E)
    tangents::RM     # (2,E)
end

function EpsilonBallGraph(x::AbstractMatrix{T}, ϵ::T) where {T<:Number}
    @assert size(x,1) == 2
    N = size(x,2)

    senders = Int[]
    receivers = Int[]

    # Brute-force O(N^2) graph construction
    for j in 2:N
        xj1 = x[1,j]
        xj2 = x[2,j]
        for i in 1:j-1
            dx1 = xj1 - x[1,i]
            dx2 = xj2 - x[2,i]
            rij = sqrt(dx1*dx1 + dx2*dx2)
            if rij < ϵ
                push!(senders, i)
                push!(receivers, j)
            end
        end
    end

    E = length(senders)

    dx = Matrix{T}(undef, 2, E)
    r = Vector{T}(undef, E)
    tangents = Matrix{T}(undef, 2, E)
    midpoints = Matrix{T}(undef, 2, E)

    @inbounds for e in 1:E
        i = senders[e]
        j = receivers[e]

        dx1 = x[1,j] - x[1,i]
        dx2 = x[2,j] - x[2,i]

        dx[1,e] = dx1
        dx[2,e] = dx2

        re = sqrt(dx1*dx1 + dx2*dx2)
        r[e] = re

        invr = one(T) / max(re, T(1e-12))
        tangents[1,e] = dx1 * invr
        tangents[2,e] = dx2 * invr

        midpoints[1,e] = (x[1,i] + x[1,j]) / 2
        midpoints[2,e] = (x[2,i] + x[2,j]) / 2
    end

    return EpsilonBallGraph(
        ϵ,
        senders,
        receivers,
        Matrix{T}(x),
        dx,
        r,
        midpoints,
        tangents,
    )
end

num_nodes(graph::EpsilonBallGraph) = size(graph.x,2)
num_edges(graph::EpsilonBallGraph) = length(graph.senders)

# ============================================================
# MEEC complex
# ============================================================

struct MEECComplex{
    I  <: Integer,
    T  <: Number,
    IV <: AbstractVector{I},
    RV <: AbstractVector{T},
    RM <: AbstractMatrix{T},
    Mask
}
    graph::EpsilonBallGraph{I,T,IV,RV,RM}
    boundary_mask::Mask
    interior_mask::Mask
    d0::SparseMatrixCSC{T,I}   # (E,N)
    m0::RV                     # (N,)
    m1::RV                     # (E,)
end

function MEECComplex(graph::EpsilonBallGraph{I,T}, boundary_mask, domain_volume, ridge = T(1e-10)) where {I,T}
    interior_mask = .!boundary_mask
    d0 = _build_incidence_matrix(graph)
    m0 = _build_virtual_volumes(graph, domain_volume, interior_mask)
    m1 = _solve_edge_areas_moment_matching(graph, interior_mask, m0, ridge)
    return MEECComplex(graph, boundary_mask, interior_mask, d0, m0, m1)
end

function _build_incidence_matrix(graph::EpsilonBallGraph{I,T}) where {I,T}
    E = num_edges(graph)
    N = num_nodes(graph)

    rows = Vector{I}(undef, 2E)
    cols = Vector{I}(undef, 2E)
    vals = Vector{T}(undef, 2E)

    k = 1
    @inbounds for e in 1:E
        rows[k] = e
        cols[k] = graph.senders[e]
        vals[k] = -one(T)
        k += 1

        rows[k] = e
        cols[k] = graph.receivers[e]
        vals[k] = one(T)
        k += 1
    end

    return sparse(rows, cols, vals, E, N)
end

function _build_virtual_volumes(graph::EpsilonBallGraph{I,T}, domain_volume, interior_mask) where {I,T}
    N = num_nodes(graph)
    κ = zeros(T, N)

    @inbounds for e in 1:num_edges(graph)
        s = graph.senders[e]
        r = graph.receivers[e]
        ϕ = _wendland_like_kernel(graph.r[e], graph.ϵ)
        κ[s] += ϕ
        κ[r] += ϕ
    end

    invκ = zeros(T, N)
    @inbounds for i in 1:N
        if interior_mask[i]
            invκ[i] = one(T) / max(κ[i], T(1e-12))
        end
    end

    invsum = max(sum(invκ), T(1e-12))

    m0 = zeros(T, N)
    @inbounds for i in 1:N
        if interior_mask[i]
            m0[i] = T(domain_volume) * invκ[i] / invsum
        end
    end

    return m0
end

function _incident_edges(graph::EpsilonBallGraph, node_idx::Int)
    inc_sender = (graph.senders .== node_idx)
    inc_receiver = (graph.receivers .== node_idx)
    inc = inc_sender .| inc_receiver
    edge_idx = findall(inc)
    return edge_idx, inc_sender, inc_receiver
end

function _solve_edge_areas_moment_matching(
    graph::EpsilonBallGraph{I,T},
    interior_mask,
    m0,
    ridge,
) where {I,T}

    E = num_edges(graph)
    N = num_nodes(graph)

    ϕ = [(_wendland_like_kernel(graph.r[e], graph.ϵ) + T(1e-12)) for e in 1:E]

    rows = Int[]
    cols = Int[]
    vals = T[]
    rhs  = T[]

    row_counter = 1

    for i in 1:N
        if !interior_mask[i]
            continue
        end

        edge_idx, _, _ = _incident_edges(graph, i)
        isempty(edge_idx) && continue

        ne = length(edge_idx)
        eta = zeros(T, 2, ne)

        for (k,e) in enumerate(edge_idx)
            if graph.senders[e] == i
                eta[:,k] .= graph.dx[:,e]
            else
                eta[:,k] .= -graph.dx[:,e]
            end
        end

        local_rows = (
            eta[1,:],
            eta[2,:],
            eta[1,:] .* eta[1,:],
            eta[1,:] .* eta[2,:],
            eta[2,:] .* eta[2,:],
        )

        local_rhs = (
            zero(T),
            zero(T),
            T(2) * m0[i],
            zero(T),
            T(2) * m0[i],
        )

        for (rr, b) in zip(local_rows, local_rhs)
            for (k,e) in enumerate(edge_idx)
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
        return ones(T, E)
    end

    B = sparse(rows, cols, vals, C, E)
    Φ = spdiagm(0 => ϕ)
    S = B * Φ * B' + spdiagm(0 => fill(ridge, C))

    λ = S \ rhs
    a = Φ * (B' * λ)

    return Vector{T}(a)
end

# ============================================================
# Gradient reconstruction
# ============================================================

struct GradientReconstruction{I<:Integer,T<:Number}
    neighbors::Vector{Vector{I}}
    wx::Vector{Vector{T}}
    wy::Vector{Vector{T}}
end

function build_gradient_reconstruction(
    graph::EpsilonBallGraph{I,T},
    interior_mask;
    ridge::T = T(1e-12),
) where {I,T}

    N = num_nodes(graph)
    neighbors = Vector{Vector{I}}(undef, N)
    wx = Vector{Vector{T}}(undef, N)
    wy = Vector{Vector{T}}(undef, N)

    for i in 1:N
        edge_idx, _, _ = _incident_edges(graph, i)

        nb = Int[]
        dxs = Vector{NTuple{2,T}}()

        for e in edge_idx
            if graph.senders[e] == i
                j = graph.receivers[e]
                dxij = (graph.dx[1,e], graph.dx[2,e])
            else
                j = graph.senders[e]
                dxij = (-graph.dx[1,e], -graph.dx[2,e])
            end
            push!(nb, j)
            push!(dxs, dxij)
        end

        pushfirst!(nb, i)
        nnei = length(nb)
        neighbors[i] = Vector{I}(nb)

        if nnei <= 1
            wx[i] = zeros(T, nnei)
            wy[i] = zeros(T, nnei)
            continue
        end

        M11 = zero(T)
        M12 = zero(T)
        M22 = zero(T)

        local_wx = zeros(T, nnei)
        local_wy = zeros(T, nnei)

        for k in 2:nnei
            dx1, dx2 = dxs[k-1]
            rij = sqrt(dx1*dx1 + dx2*dx2)
            wij = _wendland_like_kernel(rij, graph.ϵ)

            M11 += wij * dx1 * dx1
            M12 += wij * dx1 * dx2
            M22 += wij * dx2 * dx2
        end

        M11 += ridge
        M22 += ridge

        detM = M11*M22 - M12*M12
        if abs(detM) < ridge
            wx[i] = zeros(T, nnei)
            wy[i] = zeros(T, nnei)
            continue
        end

        invM11 =  M22 / detM
        invM12 = -M12 / detM
        invM22 =  M11 / detM

        sx = zero(T)
        sy = zero(T)

        for k in 2:nnei
            dx1, dx2 = dxs[k-1]
            rij = sqrt(dx1*dx1 + dx2*dx2)
            wij = _wendland_like_kernel(rij, graph.ϵ)

            b1 = wij * dx1
            b2 = wij * dx2

            gx = invM11*b1 + invM12*b2
            gy = invM12*b1 + invM22*b2

            local_wx[k] = gx
            local_wy[k] = gy

            sx += gx
            sy += gy
        end

        local_wx[1] = -sx
        local_wy[1] = -sy

        wx[i] = local_wx
        wy[i] = local_wy
    end

    return GradientReconstruction(neighbors, wx, wy)
end

function apply_scalar_gradient!(
    gx::AbstractVector{T},
    gy::AbstractVector{T},
    values::AbstractVector{T},
    gradrec::GradientReconstruction{I,T},
) where {I,T}

    N = length(gradrec.neighbors)
    @inbounds for i in 1:N
        accx = zero(T)
        accy = zero(T)

        nb = gradrec.neighbors[i]
        wx = gradrec.wx[i]
        wy = gradrec.wy[i]

        for k in eachindex(nb)
            j = nb[k]
            v = values[j]
            accx += wx[k] * v
            accy += wy[k] * v
        end

        gx[i] = accx
        gy[i] = accy
    end
    return nothing
end

# ============================================================
# DOF-level Dirichlet BC data
# ============================================================

struct DirichletBCData{T<:Number}
    is_constrained::BitMatrix   # (ndim, N)
    values::Matrix{T}           # (ndim, N)
end

function build_dirichlet_bc_data(::Type{T}, ndim::Int, N::Int) where {T<:Number}
    return DirichletBCData(falses(ndim, N), zeros(T, ndim, N))
end

function constrain_dofs!(
    bc::DirichletBCData{T},
    node_ids,
    dof_ids,
    values,
) where {T}

    nd = length(dof_ids)
    nn = length(node_ids)

    if isa(values, Number)
        @inbounds for j in 1:nn
            node = node_ids[j]
            for a in 1:nd
                dof = dof_ids[a]
                bc.is_constrained[dof, node] = true
                bc.values[dof, node] = T(values)
            end
        end

    elseif isa(values, AbstractVector)
        @assert nd == 1 "Vector values only supported for a single dof."
        @assert length(values) == nn
        dof = dof_ids[1]
        @inbounds for j in 1:nn
            node = node_ids[j]
            bc.is_constrained[dof, node] = true
            bc.values[dof, node] = values[j]
        end

    elseif isa(values, AbstractMatrix)
        @assert size(values,1) == nd
        @assert size(values,2) == nn
        @inbounds for j in 1:nn
            node = node_ids[j]
            for a in 1:nd
                dof = dof_ids[a]
                bc.is_constrained[dof, node] = true
                bc.values[dof, node] = values[a,j]
            end
        end

    else
        error("Unsupported values type in constrain_dofs!")
    end

    return bc
end

# ============================================================
# Manufactured displacement helpers
# ============================================================

function affine_displacement_field(
    x::AbstractMatrix{T},   # (2,N)
    A::AbstractMatrix{T},   # (2,2)
    b::AbstractVector{T},   # (2,)
) where {T}

    N = size(x,2)
    @assert size(x,1) == 2
    @assert size(A,1) == 2 && size(A,2) == 2
    @assert length(b) == 2

    g = zeros(T, 2, N)

    @inbounds for i in 1:N
        X = x[1,i]
        Y = x[2,i]
        g[1,i] = A[1,1]*X + A[1,2]*Y + b[1]
        g[2,i] = A[2,1]*X + A[2,2]*Y + b[2]
    end

    return g
end

function affine_zero_body_force(x::AbstractMatrix{T}) where {T}
    return zeros(T, 2, size(x,2))
end

function homogeneous_deformation_displacement(
    x::AbstractMatrix{T},   # (2,N)
    F0::AbstractMatrix{T},  # (2,2)
    c::AbstractVector{T},   # (2,)
) where {T}

    N = size(x,2)
    @assert size(x,1) == 2
    @assert size(F0,1) == 2 && size(F0,2) == 2
    @assert length(c) == 2

    g = zeros(T, 2, N)

    A11 = F0[1,1] - one(T)
    A12 = F0[1,2]
    A21 = F0[2,1]
    A22 = F0[2,2] - one(T)

    @inbounds for i in 1:N
        X = x[1,i]
        Y = x[2,i]
        g[1,i] = A11*X + A12*Y + c[1]
        g[2,i] = A21*X + A22*Y + c[2]
    end

    return g
end

function homogeneous_zero_body_force(x::AbstractMatrix{T}) where {T}
    return zeros(T, 2, size(x,2))
end

# ============================================================
# Neo-Hookean nonlinear elasticity (continuum-faithful)
# ============================================================

struct NeoHookeanProblem{I<:Integer,T<:Number}
    complex::MEECComplex
    gradrec::GradientReconstruction{I,T}
    λ::T
    μ::T
end

mutable struct NeoHookeanWorkspace{T<:Number}
    dux_dX::Vector{T}
    dux_dY::Vector{T}
    duy_dX::Vector{T}
    duy_dY::Vector{T}

    P11::Vector{T}
    P12::Vector{T}
    P21::Vector{T}
    P22::Vector{T}

    dP11_dX::Vector{T}
    dP11_dY::Vector{T}
    dP12_dX::Vector{T}
    dP12_dY::Vector{T}
    dP21_dX::Vector{T}
    dP21_dY::Vector{T}
    dP22_dX::Vector{T}
    dP22_dY::Vector{T}

    r::Vector{T}
    r_base::Vector{T}

    J::Matrix{T}
    du::Vector{T}
    U_trial::Vector{T}
end

function build_neo_hookean_workspace(problem::NeoHookeanProblem{I,T}) where {I,T}
    N = num_nodes(problem.complex.graph)
    ndof = 2N
    return NeoHookeanWorkspace(
        zeros(T,N), zeros(T,N), zeros(T,N), zeros(T,N),
        zeros(T,N), zeros(T,N), zeros(T,N), zeros(T,N),
        zeros(T,N), zeros(T,N), zeros(T,N), zeros(T,N),
        zeros(T,N), zeros(T,N), zeros(T,N), zeros(T,N),
        zeros(T,ndof),
        zeros(T,ndof),
        zeros(T,ndof,ndof),
        zeros(T,ndof),
        zeros(T,ndof),
    )
end

@inline function _uview(U::AbstractVector{T}, N::Int) where {T}
    return reshape(U, 2, N)
end

function apply_neo_hookean_operator!(
    ws::NeoHookeanWorkspace{T},
    U::AbstractVector{T},
    problem::NeoHookeanProblem{I,T},
    f::AbstractMatrix{T},         # (2,N)
    bc::DirichletBCData{T},
) where {I,T}

    complex = problem.complex
    gradrec = problem.gradrec
    λ = problem.λ
    μ = problem.μ

    N = num_nodes(complex.graph)
    @assert length(U) == 2N
    @assert size(f,1) == 2 && size(f,2) == N
    @assert size(bc.is_constrained,1) == 2 && size(bc.is_constrained,2) == N
    @assert size(bc.values,1) == 2 && size(bc.values,2) == N

    Um = _uview(U, N)
    ux = @view Um[1,:]
    uy = @view Um[2,:]

    apply_scalar_gradient!(ws.dux_dX, ws.dux_dY, ux, gradrec)
    apply_scalar_gradient!(ws.duy_dX, ws.duy_dY, uy, gradrec)

    @inbounds for i in 1:N
        F11 = one(T) + ws.dux_dX[i]
        F12 = ws.dux_dY[i]
        F21 = ws.duy_dX[i]
        F22 = one(T) + ws.duy_dY[i]

        J = F11*F22 - F12*F21
        if J <= zero(T)
            J = eps(T)
        end

        invJ = one(T) / J

        Finv11 =  F22 * invJ
        Finv12 = -F12 * invJ
        Finv21 = -F21 * invJ
        Finv22 =  F11 * invJ

        Fit11 = Finv11
        Fit12 = Finv21
        Fit21 = Finv12
        Fit22 = Finv22

        logJ = log(J)

        ws.P11[i] = μ*(F11 - Fit11) + λ*logJ*Fit11
        ws.P12[i] = μ*(F12 - Fit12) + λ*logJ*Fit12
        ws.P21[i] = μ*(F21 - Fit21) + λ*logJ*Fit21
        ws.P22[i] = μ*(F22 - Fit22) + λ*logJ*Fit22
    end

    apply_scalar_gradient!(ws.dP11_dX, ws.dP11_dY, ws.P11, gradrec)
    apply_scalar_gradient!(ws.dP12_dX, ws.dP12_dY, ws.P12, gradrec)
    apply_scalar_gradient!(ws.dP21_dX, ws.dP21_dY, ws.P21, gradrec)
    apply_scalar_gradient!(ws.dP22_dX, ws.dP22_dY, ws.P22, gradrec)

    @inbounds for i in 1:N
        divP1 = ws.dP11_dX[i] + ws.dP12_dY[i]
        divP2 = ws.dP21_dX[i] + ws.dP22_dY[i]

        ws.r[2i-1] = -divP1 - f[1,i]
        ws.r[2i]   = -divP2 - f[2,i]
    end

    @inbounds for i in 1:N
        if bc.is_constrained[1,i]
            ws.r[2i-1] = Um[1,i] - bc.values[1,i]
        end
        if bc.is_constrained[2,i]
            ws.r[2i] = Um[2,i] - bc.values[2,i]
        end
    end

    return ws.r
end

function assemble_neo_hookean_jacobian_fd!(
    ws::NeoHookeanWorkspace{T},
    U::AbstractVector{T},
    problem::NeoHookeanProblem{I,T},
    f::AbstractMatrix{T},
    bc::DirichletBCData{T};
    fd_eps::T = sqrt(eps(T)),
) where {I,T}

    ndof = length(U)

    apply_neo_hookean_operator!(ws, U, problem, f, bc)
    copyto!(ws.r_base, ws.r)

    @inbounds for j in 1:ndof
        copyto!(ws.U_trial, U)
        ws.U_trial[j] += fd_eps

        apply_neo_hookean_operator!(ws, ws.U_trial, problem, f, bc)
        @views ws.J[:,j] .= (ws.r .- ws.r_base) ./ fd_eps
    end

    copyto!(ws.r, ws.r_base)
    return ws.J
end

function solve_neo_hookean(
    problem::NeoHookeanProblem{I,T},
    f::AbstractMatrix{T},
    bc::DirichletBCData{T};
    u0 = nothing,
    max_newton::Int = 20,
    tol::T = T(1e-8),
    line_search_steps::Int = 10,
    fd_eps::T = sqrt(eps(T)),
    verbose::Bool = false,
    workspace::Union{Nothing,NeoHookeanWorkspace{T}} = nothing,
) where {I,T}

    N = num_nodes(problem.complex.graph)
    ndof = 2N
    ws = isnothing(workspace) ? build_neo_hookean_workspace(problem) : workspace

    U = isnothing(u0) ? zeros(T, ndof) : copy(u0)
    Um = _uview(U, N)

    @inbounds for i in 1:N
        if bc.is_constrained[1,i]
            Um[1,i] = bc.values[1,i]
        end
        if bc.is_constrained[2,i]
            Um[2,i] = bc.values[2,i]
        end
    end

    for k in 1:max_newton
        apply_neo_hookean_operator!(ws, U, problem, f, bc)
        rnorm = norm(ws.r)

        verbose && println("Newton iter $k, residual norm = $rnorm")

        if rnorm < tol
            return U
        end

        assemble_neo_hookean_jacobian_fd!(ws, U, problem, f, bc; fd_eps=fd_eps)
        ws.du .= -(ws.J \ ws.r)

        accepted = false
        α = one(T)

        for ls in 1:line_search_steps
            @inbounds for i in 1:ndof
                ws.U_trial[i] = U[i] + α * ws.du[i]
            end

            Utrialm = _uview(ws.U_trial, N)
            @inbounds for i in 1:N
                if bc.is_constrained[1,i]
                    Utrialm[1,i] = bc.values[1,i]
                end
                if bc.is_constrained[2,i]
                    Utrialm[2,i] = bc.values[2,i]
                end
            end

            apply_neo_hookean_operator!(ws, ws.U_trial, problem, f, bc)
            trial_norm = norm(ws.r)

            verbose && println("  line search $ls: α = $α, residual = $trial_norm")

            if trial_norm < rnorm
                copyto!(U, ws.U_trial)
                accepted = true
                break
            end

            α /= 2
        end

        if !accepted
            verbose && println("Newton line search failed; exiting.")
            return U
        end
    end

    return U
end

function solve_neo_hookean_continuation(
    problem::NeoHookeanProblem{I,T},
    f::AbstractMatrix{T},
    bc_full::DirichletBCData{T};
    nsteps::Int = 10,
    u0 = nothing,
    max_newton::Int = 20,
    tol::T = T(1e-8),
    line_search_steps::Int = 10,
    fd_eps::T = sqrt(eps(T)),
    verbose::Bool = false,
    workspace::Union{Nothing,NeoHookeanWorkspace{T}} = nothing,
) where {I,T}

    N = num_nodes(problem.complex.graph)
    ndof = 2N
    ws = isnothing(workspace) ? build_neo_hookean_workspace(problem) : workspace

    U = isnothing(u0) ? zeros(T, ndof) : copy(u0)
    bc_step = DirichletBCData(copy(bc_full.is_constrained), similar(bc_full.values))

    for step in 1:nsteps
        α = T(step) / T(nsteps)
        @. bc_step.values = α * bc_full.values

        if verbose
            println("==================================================")
            println("Continuation step $step / $nsteps, α = $α")
            println("==================================================")
        end

        U = solve_neo_hookean(
            problem,
            f,
            bc_step;
            u0 = U,
            max_newton = max_newton,
            tol = tol,
            line_search_steps = line_search_steps,
            fd_eps = fd_eps,
            verbose = verbose,
            workspace = ws,
        )
    end

    return U
end

end # module MeshFreeExteriorCalculus
