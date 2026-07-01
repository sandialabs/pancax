module MeshFreeExteriorCalculus

using LinearAlgebra

# ============================================================
# Exports
# ============================================================

export meshgrid
export make_boundary_mask

export EpsilonBallGraph
export num_nodes
export num_edges
export build_virtual_volumes

export GradientReconstruction
export build_gradient_reconstruction

export DirichletBCData
export build_dirichlet_bc_data
export constrain_dofs!
export enforce_dirichlet_values!

export NeoHookeanEnergyProblem
export NeoHookeanEnergyWorkspace
export build_neo_hookean_energy_workspace

export total_potential_energy
export energy_residual!
export energy_hessian!
export solve_neo_hookean_energy
export solve_neo_hookean_energy_continuation

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

# ============================================================
# Graph
# ============================================================

struct EpsilonBallGraph{
    I  <: Integer,
    T  <: Number
}
    ϵ::T
    senders::Vector{I}
    receivers::Vector{I}
    x::Matrix{T}          # (2,N)
    dx::Matrix{T}         # (2,E)
    r::Vector{T}          # (E,)
    midpoints::Matrix{T}  # (2,E)
    tangents::Matrix{T}   # (2,E)
end

function EpsilonBallGraph(x::AbstractMatrix{T}, ϵ::T) where {T<:Number}
    @assert size(x,1) == 2
    N = size(x,2)

    senders = Int[]
    receivers = Int[]

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
# Virtual nodal volumes
# ============================================================

function build_virtual_volumes(
    graph::EpsilonBallGraph{I,T},
    domain_volume,
    interior_mask,
) where {I,T}
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

# ============================================================
# Gradient reconstruction
# ============================================================

struct GradientReconstruction{I<:Integer,T<:Number}
    neighbors::Vector{Vector{I}}
    wx::Vector{Vector{T}}
    wy::Vector{Vector{T}}
end

function _incident_edges(graph::EpsilonBallGraph, node_idx::Int)
    inc_sender = (graph.senders .== node_idx)
    inc_receiver = (graph.receivers .== node_idx)
    inc = inc_sender .| inc_receiver
    edge_idx = findall(inc)
    return edge_idx
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
        edge_idx = _incident_edges(graph, i)

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
# Dirichlet BC data
# ============================================================

struct DirichletBCData{T<:Number}
    is_constrained::BitMatrix
    values::Matrix{T}
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
        @assert nd == 1
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
        error("Unsupported values type")
    end

    return bc
end

function enforce_dirichlet_values!(U::AbstractVector{T}, bc::DirichletBCData{T}) where {T}
    N = size(bc.values, 2)
    Um = reshape(U, 2, N)
    @inbounds for i in 1:N
        if bc.is_constrained[1,i]
            Um[1,i] = bc.values[1,i]
        end
        if bc.is_constrained[2,i]
            Um[2,i] = bc.values[2,i]
        end
    end
    return U
end

# ============================================================
# Neo-Hookean energy problem
# ============================================================

struct NeoHookeanEnergyProblem{I<:Integer,T<:Number}
    graph::EpsilonBallGraph{I,T}
    gradrec::GradientReconstruction{I,T}
    m0::Vector{T}
    λ::T
    μ::T
end

mutable struct NeoHookeanEnergyWorkspace{T<:Number}
    dux_dX::Vector{T}
    dux_dY::Vector{T}
    duy_dX::Vector{T}
    duy_dY::Vector{T}

    # stress-like first Piola at nodes
    P11::Vector{T}
    P12::Vector{T}
    P21::Vector{T}
    P22::Vector{T}

    r::Vector{T}
    H::Matrix{T}
    U_trial::Vector{T}
end

function build_neo_hookean_energy_workspace(
    problem::NeoHookeanEnergyProblem{I,T}
) where {I,T}
    N = num_nodes(problem.graph)
    ndof = 2N
    return NeoHookeanEnergyWorkspace(
        zeros(T,N), zeros(T,N), zeros(T,N), zeros(T,N),
        zeros(T,N), zeros(T,N), zeros(T,N), zeros(T,N),
        zeros(T,ndof),
        zeros(T,ndof,ndof),
        zeros(T,ndof),
    )
end

@inline function _uview(U::AbstractVector{T}, N::Int) where {T}
    reshape(U, 2, N)
end

# ------------------------------------------------------------
# Material model
# ------------------------------------------------------------

@inline function neo_hookean_energy_density(F11, F12, F21, F22, λ, μ)
    J = F11*F22 - F12*F21
    if J <= 0
        return Inf
    end
    I1 = F11*F11 + F12*F12 + F21*F21 + F22*F22
    logJ = log(J)
    return (μ/2) * (I1 - 2 - 2logJ) + (λ/2) * (logJ^2)
end

@inline function neo_hookean_piola(F11, F12, F21, F22, λ, μ)
    J = F11*F22 - F12*F21
    if J <= 0
        return (Inf, Inf, Inf, Inf)
    end

    invJ = 1 / J
    Finv11 =  F22 * invJ
    Finv12 = -F12 * invJ
    Finv21 = -F21 * invJ
    Finv22 =  F11 * invJ

    Fit11 = Finv11
    Fit12 = Finv21
    Fit21 = Finv12
    Fit22 = Finv22

    logJ = log(J)

    P11 = μ*(F11 - Fit11) + λ*logJ*Fit11
    P12 = μ*(F12 - Fit12) + λ*logJ*Fit12
    P21 = μ*(F21 - Fit21) + λ*logJ*Fit21
    P22 = μ*(F22 - Fit22) + λ*logJ*Fit22

    return P11, P12, P21, P22
end

"""
Directional tangent action:
Given F and perturbation dF, compute dP = ∂P/∂F : dF
for compressible Neo-Hookean.

Uses:
  dP = μ dF + (μ - λ log J) F^{-T} dF^T F^{-T}
       + λ tr(F^{-1} dF) F^{-T}
"""
@inline function neo_hookean_tangent_action(
    F11, F12, F21, F22,
    dF11, dF12, dF21, dF22,
    λ, μ
)
    J = F11*F22 - F12*F21
    if J <= 0
        return (Inf, Inf, Inf, Inf)
    end

    invJ = 1 / J
    Finv11 =  F22 * invJ
    Finv12 = -F12 * invJ
    Finv21 = -F21 * invJ
    Finv22 =  F11 * invJ

    Fit11 = Finv11
    Fit12 = Finv21
    Fit21 = Finv12
    Fit22 = Finv22

    logJ = log(J)

    # tr(F^{-1} dF)
    trFinvdF =
        Finv11*dF11 + Finv12*dF21 +
        Finv21*dF12 + Finv22*dF22

    # M = F^{-T} dF^T F^{-T}
    # first compute dF^T:
    # [dF11 dF21; dF12 dF22]

    # A = dF^T * F^{-T}
    A11 = dF11*Fit11 + dF21*Fit21
    A12 = dF11*Fit12 + dF21*Fit22
    A21 = dF12*Fit11 + dF22*Fit21
    A22 = dF12*Fit12 + dF22*Fit22

    # M = F^{-T} * A
    M11 = Fit11*A11 + Fit12*A21
    M12 = Fit11*A12 + Fit12*A22
    M21 = Fit21*A11 + Fit22*A21
    M22 = Fit21*A12 + Fit22*A22

    c = μ - λ*logJ

    dP11 = μ*dF11 + c*M11 + λ*trFinvdF*Fit11
    dP12 = μ*dF12 + c*M12 + λ*trFinvdF*Fit12
    dP21 = μ*dF21 + c*M21 + λ*trFinvdF*Fit21
    dP22 = μ*dF22 + c*M22 + λ*trFinvdF*Fit22

    return dP11, dP12, dP21, dP22
end

# ------------------------------------------------------------
# Energy
# ------------------------------------------------------------

function total_potential_energy(
    U::AbstractVector{T},
    problem::NeoHookeanEnergyProblem{I,T},
    f::AbstractMatrix{T},
    bc::DirichletBCData{T},
    ws::NeoHookeanEnergyWorkspace{T},
) where {I,T}

    N = num_nodes(problem.graph)
    @assert length(U) == 2N
    @assert size(f,1) == 2 && size(f,2) == N

    copyto!(ws.U_trial, U)
    enforce_dirichlet_values!(ws.U_trial, bc)

    Um = _uview(ws.U_trial, N)
    ux = @view Um[1,:]
    uy = @view Um[2,:]

    apply_scalar_gradient!(ws.dux_dX, ws.dux_dY, ux, problem.gradrec)
    apply_scalar_gradient!(ws.duy_dX, ws.duy_dY, uy, problem.gradrec)

    Π = zero(T)

    @inbounds for i in 1:N
        F11 = one(T) + ws.dux_dX[i]
        F12 = ws.dux_dY[i]
        F21 = ws.duy_dX[i]
        F22 = one(T) + ws.duy_dY[i]

        W = neo_hookean_energy_density(F11, F12, F21, F22, problem.λ, problem.μ)
        if !isfinite(W)
            return T(Inf)
        end

        Π += problem.m0[i] * W
        Π -= problem.m0[i] * (f[1,i]*Um[1,i] + f[2,i]*Um[2,i])
    end

    return Π
end

# ------------------------------------------------------------
# Analytic residual
# ------------------------------------------------------------

"""
Analytic energy gradient:
    R(U) = ∂Π/∂U

Since F_i depends linearly on nodal dofs through reconstruction weights,
the residual can be assembled directly from nodal Piola stresses.
"""
function energy_residual!(
    r::AbstractVector{T},
    U::AbstractVector{T},
    problem::NeoHookeanEnergyProblem{I,T},
    f::AbstractMatrix{T},
    bc::DirichletBCData{T},
    ws::NeoHookeanEnergyWorkspace{T},
) where {I,T}

    N = num_nodes(problem.graph)
    fill!(r, zero(T))

    copyto!(ws.U_trial, U)
    enforce_dirichlet_values!(ws.U_trial, bc)

    Um = _uview(ws.U_trial, N)
    ux = @view Um[1,:]
    uy = @view Um[2,:]

    apply_scalar_gradient!(ws.dux_dX, ws.dux_dY, ux, problem.gradrec)
    apply_scalar_gradient!(ws.duy_dX, ws.duy_dY, uy, problem.gradrec)

    # nodal Piola stresses
    @inbounds for i in 1:N
        F11 = one(T) + ws.dux_dX[i]
        F12 = ws.dux_dY[i]
        F21 = ws.duy_dX[i]
        F22 = one(T) + ws.duy_dY[i]

        P11, P12, P21, P22 = neo_hookean_piola(F11, F12, F21, F22, problem.λ, problem.μ)

        ws.P11[i] = P11
        ws.P12[i] = P12
        ws.P21[i] = P21
        ws.P22[i] = P22
    end

    # internal virtual work derivative:
    # Π_int = Σ_i m_i W(F_i(U))
    #
    # Since F_i depends linearly on ux_j, uy_j through wx, wy,
    # residual contributions are:
    #
    # dΠ/dux_j = Σ_i m_i [ P11_i * wx_ij + P12_i * wy_ij ]
    # dΠ/duy_j = Σ_i m_i [ P21_i * wx_ij + P22_i * wy_ij ]
    #
    @inbounds for i in 1:N
        mi = problem.m0[i]
        nb = problem.gradrec.neighbors[i]
        wx = problem.gradrec.wx[i]
        wy = problem.gradrec.wy[i]

        P11 = ws.P11[i]
        P12 = ws.P12[i]
        P21 = ws.P21[i]
        P22 = ws.P22[i]

        for k in eachindex(nb)
            j = nb[k]
            wX = wx[k]
            wY = wy[k]

            r[2j-1] += mi * (P11*wX + P12*wY)
            r[2j]   += mi * (P21*wX + P22*wY)
        end
    end

    # external force contribution
    @inbounds for j in 1:N
        r[2j-1] -= problem.m0[j] * f[1,j]
        r[2j]   -= problem.m0[j] * f[2,j]
    end

    # Dirichlet overwrite
    @inbounds for j in 1:N
        if bc.is_constrained[1,j]
            r[2j-1] = U[2j-1] - bc.values[1,j]
        end
        if bc.is_constrained[2,j]
            r[2j] = U[2j] - bc.values[2,j]
        end
    end

    return r
end

# ------------------------------------------------------------
# Analytic Hessian
# ------------------------------------------------------------

"""
Analytic Hessian of the discrete energy.

For each node i, and each pair of nodal dofs j, l in the reconstruction stencil,
we use the material tangent action:
    dP = A(F_i) : dF

and accumulate:
    H[a_j, b_l] += m_i * (∂P / ∂u_{b_l}) contracted with reconstruction of dof a_j.
"""
function energy_hessian!(
    H::AbstractMatrix{T},
    U::AbstractVector{T},
    problem::NeoHookeanEnergyProblem{I,T},
    f::AbstractMatrix{T},
    bc::DirichletBCData{T},
    ws::NeoHookeanEnergyWorkspace{T},
) where {I,T}

    N = num_nodes(problem.graph)
    fill!(H, zero(T))

    copyto!(ws.U_trial, U)
    enforce_dirichlet_values!(ws.U_trial, bc)

    Um = _uview(ws.U_trial, N)
    ux = @view Um[1,:]
    uy = @view Um[2,:]

    apply_scalar_gradient!(ws.dux_dX, ws.dux_dY, ux, problem.gradrec)
    apply_scalar_gradient!(ws.duy_dX, ws.duy_dY, uy, problem.gradrec)

    @inbounds for i in 1:N
        F11 = one(T) + ws.dux_dX[i]
        F12 = ws.dux_dY[i]
        F21 = ws.duy_dX[i]
        F22 = one(T) + ws.duy_dY[i]

        mi = problem.m0[i]
        nb = problem.gradrec.neighbors[i]
        wx = problem.gradrec.wx[i]
        wy = problem.gradrec.wy[i]

        #
        # For each source dof (node l, component b), dF is:
        #
        # if b = x:
        #   dF = [ wx_il  wy_il
        #          0      0    ]
        #
        # if b = y:
        #   dF = [ 0      0
        #          wx_il  wy_il ]
        #
        # Apply tangent, then contract into target dof (node j, component a):
        #
        # if a = x:
        #   contribution = m_i [ dP11 * wx_ij + dP12 * wy_ij ]
        #
        # if a = y:
        #   contribution = m_i [ dP21 * wx_ij + dP22 * wy_ij ]
        #
        for kl in eachindex(nb)
            l = nb[kl]
            wXl = wx[kl]
            wYl = wy[kl]

            # source dof: x-component at node l
            dP11x, dP12x, dP21x, dP22x =
                neo_hookean_tangent_action(
                    F11, F12, F21, F22,
                    wXl, wYl, zero(T), zero(T),
                    problem.λ, problem.μ
                )

            # source dof: y-component at node l
            dP11y, dP12y, dP21y, dP22y =
                neo_hookean_tangent_action(
                    F11, F12, F21, F22,
                    zero(T), zero(T), wXl, wYl,
                    problem.λ, problem.μ
                )

            colx = 2l - 1
            coly = 2l

            for kj in eachindex(nb)
                j = nb[kj]
                wXj = wx[kj]
                wYj = wy[kj]

                rowx = 2j - 1
                rowy = 2j

                # target residual x-component
                H[rowx, colx] += mi * (dP11x*wXj + dP12x*wYj)
                H[rowx, coly] += mi * (dP11y*wXj + dP12y*wYj)

                # target residual y-component
                H[rowy, colx] += mi * (dP21x*wXj + dP22x*wYj)
                H[rowy, coly] += mi * (dP21y*wXj + dP22y*wYj)
            end
        end
    end

    # Dirichlet overwrite: identity rows/cols
    @inbounds for j in 1:N
        if bc.is_constrained[1,j]
            idx = 2j - 1
            H[idx,:] .= zero(T)
            H[:,idx] .= zero(T)
            H[idx,idx] = one(T)
        end
        if bc.is_constrained[2,j]
            idx = 2j
            H[idx,:] .= zero(T)
            H[:,idx] .= zero(T)
            H[idx,idx] = one(T)
        end
    end

    return H
end

# ------------------------------------------------------------
# Newton solve
# ------------------------------------------------------------

function solve_neo_hookean_energy(
    problem::NeoHookeanEnergyProblem{I,T},
    f::AbstractMatrix{T},
    bc::DirichletBCData{T};
    u0 = nothing,
    max_newton::Int = 20,
    tol::T = T(1e-8),
    line_search_steps::Int = 12,
    verbose::Bool = false,
    workspace::Union{Nothing,NeoHookeanEnergyWorkspace{T}} = nothing,
) where {I,T}

    N = num_nodes(problem.graph)
    ndof = 2N
    ws = isnothing(workspace) ? build_neo_hookean_energy_workspace(problem) : workspace

    U = isnothing(u0) ? zeros(T, ndof) : copy(u0)
    enforce_dirichlet_values!(U, bc)

    for k in 1:max_newton
        energy_residual!(ws.r, U, problem, f, bc, ws)
        rnorm = norm(ws.r)

        if verbose
            Π = total_potential_energy(U, problem, f, bc, ws)
            println("Newton iter $k, energy = $Π, residual norm = $rnorm")
        end

        if rnorm < tol
            return U
        end

        energy_hessian!(ws.H, U, problem, f, bc, ws)
        ΔU = -(ws.H \ ws.r)

        accepted = false
        α = one(T)
        Π0 = total_potential_energy(U, problem, f, bc, ws)

        for ls in 1:line_search_steps
            @inbounds for i in 1:ndof
                ws.U_trial[i] = U[i] + α * ΔU[i]
            end

            enforce_dirichlet_values!(ws.U_trial, bc)
            Πtrial = total_potential_energy(ws.U_trial, problem, f, bc, ws)

            if verbose
                println("  line search $ls: α = $α, energy = $Πtrial")
            end

            if isfinite(Πtrial) && Πtrial < Π0
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

# ------------------------------------------------------------
# Continuation
# ------------------------------------------------------------

function solve_neo_hookean_energy_continuation(
    problem::NeoHookeanEnergyProblem{I,T},
    f::AbstractMatrix{T},
    bc_full::DirichletBCData{T};
    nsteps::Int = 10,
    u0 = nothing,
    max_newton::Int = 20,
    tol::T = T(1e-8),
    line_search_steps::Int = 12,
    verbose::Bool = false,
    workspace::Union{Nothing,NeoHookeanEnergyWorkspace{T}} = nothing,
) where {I,T}

    N = num_nodes(problem.graph)
    ndof = 2N
    ws = isnothing(workspace) ? build_neo_hookean_energy_workspace(problem) : workspace

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

        U = solve_neo_hookean_energy(
            problem,
            f,
            bc_step;
            u0 = U,
            max_newton = max_newton,
            tol = tol,
            line_search_steps = line_search_steps,
            verbose = verbose,
            workspace = ws,
        )
    end

    return U
end

end # module MeshfreeHyperelasticEnergyAnalytic