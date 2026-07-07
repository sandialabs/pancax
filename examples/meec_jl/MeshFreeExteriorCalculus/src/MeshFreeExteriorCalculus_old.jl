# module MeshFreeExteriorCalculus

# using LinearAlgebra
# using SparseArrays

# # ============================================================
# # Exports
# # ============================================================

# export meshgrid
# export EpsilonBallGraph
# export MEECComplex

# export AbstractFluxModel
# export ZeroFluxModel
# export ConstantAdvectionFluxModel
# export NonlinearSolveWorkspace
# export build_nonlinear_workspace
# export assemble_residual!
# export assemble_jacobian!
# export solve_dirichlet_nonlinear

# export GradientReconstruction
# export build_gradient_reconstruction

# export LinearElasticityProblem
# export LinearElasticityWorkspace
# export build_linear_elasticity_workspace
# export apply_linear_elasticity_operator!
# export assemble_linear_elasticity_matrix
# export solve_linear_elasticity
# export affine_displacement_field
# export affine_zero_body_force

# export NeoHookeanProblem
# export NeoHookeanWorkspace
# export build_neo_hookean_workspace
# export apply_neo_hookean_operator!
# export assemble_neo_hookean_jacobian_fd!
# export solve_neo_hookean
# export homogeneous_deformation_displacement
# export homogeneous_zero_body_force

# # ============================================================
# # Utilities
# # ============================================================

# function meshgrid(
#     nx::Int, ny::Int;
#     xmin = 0.0, xmax = 1.0,
#     ymin = 0.0, ymax = 1.0
# )
#     x = LinRange(xmin, xmax, nx)
#     y = LinRange(ymin, ymax, ny)
#     return meshgrid(x, y)
# end

# function meshgrid(xin::LinRange{T,I}, yin::LinRange{T,I}) where {T<:Number,I<:Integer}
#     nx = length(xin)
#     ny = length(yin)
#     xout = zeros(T, ny, nx)
#     yout = zeros(T, ny, nx)
#     for jx in 1:nx
#         for iy in 1:ny
#             xout[iy, jx] = xin[jx]
#             yout[iy, jx] = yin[iy]
#         end
#     end
#     return (x=xout, y=yout)
# end

# function _wendland_like_kernel(r::T, ϵ::T) where {T<:Number}
#     z = one(T) - r / ϵ
#     return z > zero(T) ? z*z : zero(T)
# end

# # ============================================================
# # Graph
# # ============================================================

# struct EpsilonBallGraph{
#     I  <: Integer,
#     T  <: Number,
#     IV <: AbstractVector{I},
#     RV <: AbstractVector{T},
#     RM <: AbstractMatrix{T}
# }
#     ϵ::T
#     senders::IV     # (E,)
#     receivers::IV   # (E,)
#     x::RM           # (N,2)
#     dx::RM          # (E,2)
#     r::RV           # (E,)
#     midpoints::RM   # (E,2)
#     tangents::RM    # (E,2)
# end

# function EpsilonBallGraph(x::Matrix{T}, ϵ::T) where {T<:Number}
#     @assert size(x,2) == 2
#     N = size(x,1)

#     diff = [x[j,k] - x[i,k] for i in axes(x,1), j in axes(x,1), k in axes(x,2)]
#     dist = dropdims(sqrt.(sum(abs2, diff; dims=3)); dims=3)

#     ii = Int[]
#     jj = Int[]
#     for j in 2:N
#         for i in 1:j-1
#             push!(ii, i)
#             push!(jj, j)
#         end
#     end

#     senders = Int[]
#     receivers = Int[]
#     for k in eachindex(ii)
#         i = ii[k]
#         j = jj[k]
#         if dist[i,j] < ϵ
#             push!(senders, i)
#             push!(receivers, j)
#         end
#     end

#     E = length(senders)

#     dx = Matrix{T}(undef, E, 2)
#     for e in 1:E
#         dx[e,:] .= x[receivers[e],:] .- x[senders[e],:]
#     end

#     r = vec(dropdims(sqrt.(sum(abs2, dx; dims=2)); dims=2))

#     tangents = similar(dx)
#     for e in 1:E
#         tangents[e,:] .= dx[e,:] ./ max(r[e], T(1e-12))
#     end

#     midpoints = Matrix{T}(undef, E, 2)
#     for e in 1:E
#         midpoints[e,:] .= 0.5 .* (x[senders[e],:] + x[receivers[e],:])
#     end

#     return EpsilonBallGraph(
#         ϵ,
#         senders,
#         receivers,
#         x,
#         dx,
#         r,
#         midpoints,
#         tangents,
#     )
# end

# num_edges(graph::EpsilonBallGraph) = length(graph.senders)
# num_nodes(graph::EpsilonBallGraph) = size(graph.x, 1)

# # ============================================================
# # MEEC complex
# # ============================================================

# struct MEECComplex{
#     I  <: Integer,
#     T  <: Number,
#     IV <: AbstractVector{I},
#     RV <: AbstractVector{T},
#     RM <: AbstractMatrix{T},
#     Mask
# }
#     graph::EpsilonBallGraph{I,T,IV,RV,RM}
#     boundary_mask::Mask
#     interior_mask::Mask
#     d0::RM
#     m0::RV
#     m1::RV
# end

# function MEECComplex(graph, boundary_mask, domain_volume, ridge = 1e-10)
#     interior_mask = .!boundary_mask
#     d0 = _build_incidence_matrix(graph)
#     m0 = _build_virtual_volumes(graph, domain_volume, interior_mask)
#     m1 = _solve_edge_areas_moment_matching(graph, interior_mask, m0, ridge)
#     return MEECComplex(graph, boundary_mask, interior_mask, d0, m0, m1)
# end

# function _build_incidence_matrix(graph::EpsilonBallGraph)
#     E = num_edges(graph)
#     N = num_nodes(graph)
#     T = eltype(graph.r)
#     d0 = zeros(T, E, N)
#     for e in 1:E
#         d0[e, graph.senders[e]] = -one(T)
#         d0[e, graph.receivers[e]] = one(T)
#     end
#     return d0
# end

# function _build_virtual_volumes(graph::EpsilonBallGraph, domain_volume, interior_mask)
#     N = size(graph.x, 1)
#     T = eltype(graph.r)

#     ϕ_e = _wendland_like_kernel.(graph.r, (graph.ϵ,))
#     κ = zeros(T, N)

#     for e in eachindex(graph.senders)
#         s = graph.senders[e]
#         r = graph.receivers[e]
#         ϕ = ϕ_e[e]
#         κ[s] += ϕ
#         κ[r] += ϕ
#     end

#     inv_κ = zeros(T, N)
#     for i in eachindex(interior_mask)
#         if interior_mask[i]
#             inv_κ[i] = one(T) / max(κ[i], T(1e-12))
#         end
#     end

#     inv_sum = max(sum(inv_κ), T(1e-12))

#     m = zeros(T, N)
#     for i in eachindex(interior_mask)
#         if interior_mask[i]
#             m[i] = T(domain_volume) * inv_κ[i] / inv_sum
#         end
#     end
#     return m
# end

# function _incident_edges(graph::EpsilonBallGraph, node_idx::Int)
#     inc_sender = (graph.senders .== node_idx)
#     inc_receiver = (graph.receivers .== node_idx)
#     inc = inc_sender .| inc_receiver
#     edge_idx = findall(inc)
#     return edge_idx, inc_sender, inc_receiver
# end

# function _solve_edge_areas_moment_matching(
#     graph::EpsilonBallGraph,
#     interior_mask,
#     m0,
#     ridge
# )
#     E = length(graph.senders)
#     N = size(graph.x, 1)
#     T = Float64

#     ϕ = [(max(1.0 - r / graph.ϵ, 0.0)^2 + 1e-12) for r in graph.r]

#     rows = Int[]
#     cols = Int[]
#     vals = T[]
#     rhs  = T[]

#     row_counter = 1

#     for i in 1:N
#         if !interior_mask[i]
#             continue
#         end

#         edge_idx, _, _ = _incident_edges(graph, i)
#         isempty(edge_idx) && continue

#         ne = length(edge_idx)
#         eta = zeros(T, ne, 2)

#         for (k, e) in enumerate(edge_idx)
#             if graph.senders[e] == i
#                 eta[k,:] .= graph.dx[e,:]
#             else
#                 eta[k,:] .= -graph.dx[e,:]
#             end
#         end

#         local_rows = (
#             eta[:,1],
#             eta[:,2],
#             eta[:,1] .* eta[:,1],
#             eta[:,1] .* eta[:,2],
#             eta[:,2] .* eta[:,2],
#         )

#         local_rhs = (
#             0.0,
#             0.0,
#             2.0 * m0[i],
#             0.0,
#             2.0 * m0[i],
#         )

#         for (rr, b) in zip(local_rows, local_rhs)
#             for (k, e) in enumerate(edge_idx)
#                 push!(rows, row_counter)
#                 push!(cols, e)
#                 push!(vals, rr[k])
#             end
#             push!(rhs, b)
#             row_counter += 1
#         end
#     end

#     C = row_counter - 1
#     if C == 0
#         return ones(T, E)
#     end

#     B = sparse(rows, cols, vals, C, E)
#     Φ = spdiagm(0 => ϕ)
#     S = B * Φ * B' + ridge * I(C)
#     λ = S \ rhs
#     a = Φ * (B' * λ)
#     return a
# end

# # ============================================================
# # Scalar nonlinear flux models
# # ============================================================

# abstract type AbstractFluxModel end

# struct ZeroFluxModel <: AbstractFluxModel end

# struct ConstantAdvectionFluxModel{T<:Number} <: AbstractFluxModel
#     vx::T
#     vy::T
# end

# mutable struct NonlinearSolveWorkspace{
#     T<:Number,
#     I<:Integer,
#     RV<:AbstractVector{T},
#     SM<:SparseMatrixCSC{T,I}
# }
#     r::RV
#     du::RV
#     u_trial::RV
#     r_trial::RV
#     flux::RV
#     dflux_duL::RV
#     dflux_duR::RV
#     J::SM
# end

# function build_nonlinear_workspace(complex::MEECComplex{I,T}) where {I,T}
#     graph = complex.graph
#     N = num_nodes(graph)
#     E = num_edges(graph)

#     row_idx = Int[]
#     col_idx = Int[]
#     sizehint!(row_idx, N + 4E)
#     sizehint!(col_idx, N + 4E)

#     for i in 1:N
#         push!(row_idx, i)
#         push!(col_idx, i)
#     end

#     for e in 1:E
#         i = graph.senders[e]
#         j = graph.receivers[e]
#         push!(row_idx, i); push!(col_idx, i)
#         push!(row_idx, i); push!(col_idx, j)
#         push!(row_idx, j); push!(col_idx, i)
#         push!(row_idx, j); push!(col_idx, j)
#     end

#     J = sparse(row_idx, col_idx, ones(T, length(row_idx)), N, N)
#     rv_state = zeros(T, N)
#     rv_edge = zeros(T, E)

#     return NonlinearSolveWorkspace(
#         rv_state,
#         similar(rv_state),
#         similar(rv_state),
#         similar(rv_state),
#         rv_edge,
#         similar(rv_edge),
#         similar(rv_edge),
#         J,
#     )
# end

# function evaluate_flux_and_partials!(
#     flux::AbstractVector{T},
#     dflux_duL::AbstractVector{T},
#     dflux_duR::AbstractVector{T},
#     ::ZeroFluxModel,
#     u::AbstractVector{T},
#     complex::MEECComplex,
#     μ = nothing,
# ) where {T}
#     fill!(flux, zero(T))
#     fill!(dflux_duL, zero(T))
#     fill!(dflux_duR, zero(T))
#     return nothing
# end

# function evaluate_flux_and_partials!(
#     flux::AbstractVector{T},
#     dflux_duL::AbstractVector{T},
#     dflux_duR::AbstractVector{T},
#     model::ConstantAdvectionFluxModel{T},
#     u::AbstractVector{T},
#     complex::MEECComplex,
#     μ = nothing,
# ) where {T}
#     graph = complex.graph
#     E = num_edges(graph)
#     @inbounds for e in 1:E
#         i = graph.senders[e]
#         j = graph.receivers[e]
#         ubar = (u[i] + u[j]) / 2
#         vt = model.vx * graph.tangents[e,1] + model.vy * graph.tangents[e,2]
#         flux[e] = -vt * ubar * graph.r[e]
#         dval = -vt * graph.r[e] / 2
#         dflux_duL[e] = dval
#         dflux_duR[e] = dval
#     end
#     return nothing
# end

# function assemble_residual!(
#     r::AbstractVector{T},
#     flux::AbstractVector{T},
#     dflux_duL::AbstractVector{T},
#     dflux_duR::AbstractVector{T},
#     model::AbstractFluxModel,
#     u::AbstractVector{T},
#     complex::MEECComplex,
#     f::AbstractVector{T},
#     g::AbstractVector{T};
#     ε::T,
#     μ = nothing,
# ) where {T}

#     graph = complex.graph
#     E = num_edges(graph)
#     N = num_nodes(graph)

#     fill!(r, zero(T))
#     evaluate_flux_and_partials!(flux, dflux_duL, dflux_duR, model, u, complex, μ)

#     @inbounds for e in 1:E
#         i = graph.senders[e]
#         j = graph.receivers[e]
#         ae = complex.m1[e]
#         du_edge = u[j] - u[i]
#         q = ε * ae * du_edge + ae * flux[e]
#         r[i] -= q
#         r[j] += q
#     end

#     @inbounds for i in 1:N
#         r[i] -= complex.m0[i] * f[i]
#     end

#     @inbounds for i in 1:N
#         if complex.boundary_mask[i]
#             r[i] = u[i] - g[i]
#         end
#     end

#     return r
# end

# function assemble_jacobian!(
#     ws::NonlinearSolveWorkspace{T,I},
#     model::AbstractFluxModel,
#     u::AbstractVector{T},
#     complex::MEECComplex,
#     f::AbstractVector{T},
#     g::AbstractVector{T};
#     ε::T,
#     μ = nothing,
# ) where {T,I}

#     graph = complex.graph
#     E = num_edges(graph)
#     N = num_nodes(graph)

#     evaluate_flux_and_partials!(
#         ws.flux, ws.dflux_duL, ws.dflux_duR, model, u, complex, μ
#     )

#     J = ws.J
#     fill!(nonzeros(J), zero(T))

#     @inbounds for e in 1:E
#         i = graph.senders[e]
#         j = graph.receivers[e]
#         ae = complex.m1[e]
#         α = ws.dflux_duL[e]
#         β = ws.dflux_duR[e]

#         J[i,i] += ε * ae
#         J[i,j] += -ε * ae
#         J[j,i] += -ε * ae
#         J[j,j] += ε * ae

#         J[i,i] += -ae * α
#         J[i,j] += -ae * β
#         J[j,i] += ae * α
#         J[j,j] += ae * β
#     end

#     @inbounds for i in 1:N
#         if complex.boundary_mask[i]
#             for j in 1:N
#                 J[i,j] = zero(T)
#             end
#             J[i,i] = one(T)
#         end
#     end

#     return J
# end

# function solve_dirichlet_nonlinear(
#     model::AbstractFluxModel,
#     complex::MEECComplex{I,T},
#     f::AbstractVector{T},
#     g::AbstractVector{T};
#     ε::T,
#     μ = nothing,
#     u0 = nothing,
#     max_newton::Int = 12,
#     tol::T = T(1e-8),
#     line_search_steps::Int = 8,
#     verbose::Bool = false,
#     workspace::Union{Nothing,NonlinearSolveWorkspace{T,I}} = nothing,
# ) where {I,T}

#     graph = complex.graph
#     N = num_nodes(graph)
#     ws = isnothing(workspace) ? build_nonlinear_workspace(complex) : workspace

#     u = isnothing(u0) ? similar(g) : copy(u0)
#     if isnothing(u0)
#         copyto!(u, g)
#     end

#     @inbounds for i in 1:N
#         if complex.boundary_mask[i]
#             u[i] = g[i]
#         end
#     end

#     for k in 1:max_newton
#         assemble_residual!(ws.r, ws.flux, ws.dflux_duL, ws.dflux_duR, model, u, complex, f, g; ε=ε, μ=μ)
#         rnorm = norm(ws.r)
#         verbose && println("Newton iter $k, residual norm = $rnorm")
#         if rnorm < tol
#             return u
#         end

#         J = assemble_jacobian!(ws, model, u, complex, f, g; ε=ε, μ=μ)
#         ws.du .= -(J \ ws.r)

#         accepted = false
#         αls = one(T)

#         for ls in 1:line_search_steps
#             @inbounds for i in 1:N
#                 ws.u_trial[i] = u[i] + αls * ws.du[i]
#                 if complex.boundary_mask[i]
#                     ws.u_trial[i] = g[i]
#                 end
#             end

#             assemble_residual!(ws.r_trial, ws.flux, ws.dflux_duL, ws.dflux_duR, model, ws.u_trial, complex, f, g; ε=ε, μ=μ)
#             trial_norm = norm(ws.r_trial)

#             verbose && println("  line search $ls: α = $αls, residual = $trial_norm")

#             if trial_norm < rnorm
#                 copyto!(u, ws.u_trial)
#                 accepted = true
#                 break
#             end

#             αls /= 2
#         end

#         if !accepted
#             verbose && println("Newton line search failed; exiting.")
#             return u
#         end
#     end

#     return u
# end

# # ============================================================
# # Gradient reconstruction
# # ============================================================

# struct GradientReconstruction{I<:Integer,T<:Number}
#     neighbors::Vector{Vector{I}}
#     wx::Vector{Vector{T}}
#     wy::Vector{Vector{T}}
# end

# function build_gradient_reconstruction(
#     graph::EpsilonBallGraph{I,T},
#     interior_mask;
#     ridge::T = T(1e-12),
# ) where {I,T}

#     N = num_nodes(graph)
#     neighbors = Vector{Vector{I}}(undef, N)
#     wx = Vector{Vector{T}}(undef, N)
#     wy = Vector{Vector{T}}(undef, N)

#     for i in 1:N
#         edge_idx, _, _ = _incident_edges(graph, i)

#         nb = Int[]
#         dxs = Vector{NTuple{2,T}}()

#         for e in edge_idx
#             if graph.senders[e] == i
#                 j = graph.receivers[e]
#                 dxij = (graph.dx[e,1], graph.dx[e,2])
#             else
#                 j = graph.senders[e]
#                 dxij = (-graph.dx[e,1], -graph.dx[e,2])
#             end
#             push!(nb, j)
#             push!(dxs, dxij)
#         end

#         pushfirst!(nb, i)
#         nnei = length(nb)
#         neighbors[i] = Vector{I}(nb)

#         if nnei <= 1
#             wx[i] = zeros(T, nnei)
#             wy[i] = zeros(T, nnei)
#             continue
#         end

#         M11 = zero(T)
#         M12 = zero(T)
#         M22 = zero(T)

#         local_wx = zeros(T, nnei)
#         local_wy = zeros(T, nnei)

#         for k in 2:nnei
#             dx1, dx2 = dxs[k-1]
#             rij = sqrt(dx1*dx1 + dx2*dx2)
#             wij = _wendland_like_kernel(rij, graph.ϵ)
#             M11 += wij * dx1 * dx1
#             M12 += wij * dx1 * dx2
#             M22 += wij * dx2 * dx2
#         end

#         M11 += ridge
#         M22 += ridge

#         detM = M11*M22 - M12*M12
#         if abs(detM) < ridge
#             wx[i] = zeros(T, nnei)
#             wy[i] = zeros(T, nnei)
#             continue
#         end

#         invM11 =  M22 / detM
#         invM12 = -M12 / detM
#         invM22 =  M11 / detM

#         sx = zero(T)
#         sy = zero(T)

#         for k in 2:nnei
#             dx1, dx2 = dxs[k-1]
#             rij = sqrt(dx1*dx1 + dx2*dx2)
#             wij = _wendland_like_kernel(rij, graph.ϵ)

#             b1 = wij * dx1
#             b2 = wij * dx2

#             gx = invM11*b1 + invM12*b2
#             gy = invM12*b1 + invM22*b2

#             local_wx[k] = gx
#             local_wy[k] = gy

#             sx += gx
#             sy += gy
#         end

#         local_wx[1] = -sx
#         local_wy[1] = -sy

#         wx[i] = local_wx
#         wy[i] = local_wy
#     end

#     return GradientReconstruction(neighbors, wx, wy)
# end

# function apply_scalar_gradient!(
#     gx::AbstractVector{T},
#     gy::AbstractVector{T},
#     values::AbstractVector{T},
#     gradrec::GradientReconstruction{I,T},
# ) where {I,T}
#     N = length(gradrec.neighbors)
#     @inbounds for i in 1:N
#         accx = zero(T)
#         accy = zero(T)

#         nb = gradrec.neighbors[i]
#         wx = gradrec.wx[i]
#         wy = gradrec.wy[i]

#         for k in eachindex(nb)
#             j = nb[k]
#             v = values[j]
#             accx += wx[k] * v
#             accy += wy[k] * v
#         end

#         gx[i] = accx
#         gy[i] = accy
#     end
#     return nothing
# end

# # ============================================================
# # Linear elasticity
# # ============================================================

# struct LinearElasticityProblem{I<:Integer,T<:Number}
#     complex::MEECComplex
#     gradrec::GradientReconstruction{I,T}
#     λ::T
#     μ::T
# end

# mutable struct LinearElasticityWorkspace{T<:Number}
#     dux_dx::Vector{T}
#     dux_dy::Vector{T}
#     duy_dx::Vector{T}
#     duy_dy::Vector{T}

#     σ11::Vector{T}
#     σ12::Vector{T}
#     σ21::Vector{T}
#     σ22::Vector{T}

#     dσ11_dx::Vector{T}
#     dσ11_dy::Vector{T}
#     dσ12_dx::Vector{T}
#     dσ12_dy::Vector{T}
#     dσ21_dx::Vector{T}
#     dσ21_dy::Vector{T}
#     dσ22_dx::Vector{T}
#     dσ22_dy::Vector{T}

#     r::Vector{T}
# end

# function build_linear_elasticity_workspace(problem::LinearElasticityProblem{I,T}) where {I,T}
#     N = num_nodes(problem.complex.graph)
#     return LinearElasticityWorkspace(
#         zeros(T,N), zeros(T,N), zeros(T,N), zeros(T,N),
#         zeros(T,N), zeros(T,N), zeros(T,N), zeros(T,N),
#         zeros(T,N), zeros(T,N), zeros(T,N), zeros(T,N),
#         zeros(T,N), zeros(T,N), zeros(T,N), zeros(T,N),
#         zeros(T,2N),
#     )
# end

# function apply_linear_elasticity_operator!(
#     ws::LinearElasticityWorkspace{T},
#     U::AbstractVector{T},
#     problem::LinearElasticityProblem{I,T},
#     f::AbstractMatrix{T},
#     g::AbstractMatrix{T},
# ) where {I,T}

#     complex = problem.complex
#     gradrec = problem.gradrec
#     λ = problem.λ
#     μ = problem.μ

#     N = num_nodes(complex.graph)
#     @assert length(U) == 2N
#     @assert size(f,1) == N && size(f,2) == 2
#     @assert size(g,1) == N && size(g,2) == 2

#     ux = @view U[1:2:2N]
#     uy = @view U[2:2:2N]

#     apply_scalar_gradient!(ws.dux_dx, ws.dux_dy, ux, gradrec)
#     apply_scalar_gradient!(ws.duy_dx, ws.duy_dy, uy, gradrec)

#     @inbounds for i in 1:N
#         ε11 = ws.dux_dx[i]
#         ε22 = ws.duy_dy[i]
#         ε12 = (ws.dux_dy[i] + ws.duy_dx[i]) / 2
#         trε = ε11 + ε22

#         ws.σ11[i] = 2μ * ε11 + λ * trε
#         ws.σ22[i] = 2μ * ε22 + λ * trε
#         ws.σ12[i] = 2μ * ε12
#         ws.σ21[i] = ws.σ12[i]
#     end

#     apply_scalar_gradient!(ws.dσ11_dx, ws.dσ11_dy, ws.σ11, gradrec)
#     apply_scalar_gradient!(ws.dσ12_dx, ws.dσ12_dy, ws.σ12, gradrec)
#     apply_scalar_gradient!(ws.dσ21_dx, ws.dσ21_dy, ws.σ21, gradrec)
#     apply_scalar_gradient!(ws.dσ22_dx, ws.dσ22_dy, ws.σ22, gradrec)

#     @inbounds for i in 1:N
#         divσ1 = ws.dσ11_dx[i] + ws.dσ12_dy[i]
#         divσ2 = ws.dσ21_dx[i] + ws.dσ22_dy[i]

#         ws.r[2i-1] = -divσ1 - f[i,1]
#         ws.r[2i]   = -divσ2 - f[i,2]
#     end

#     @inbounds for i in 1:N
#         if complex.boundary_mask[i]
#             ws.r[2i-1] = ux[i] - g[i,1]
#             ws.r[2i]   = uy[i] - g[i,2]
#         end
#     end

#     return ws.r
# end

# function assemble_linear_elasticity_matrix(
#     problem::LinearElasticityProblem{I,T},
#     f::AbstractMatrix{T},
#     g::AbstractMatrix{T};
#     workspace::Union{Nothing,LinearElasticityWorkspace{T}} = nothing,
# ) where {I,T}

#     N = num_nodes(problem.complex.graph)
#     ndof = 2N
#     ws = isnothing(workspace) ? build_linear_elasticity_workspace(problem) : workspace

#     A = Matrix{T}(undef, ndof, ndof)
#     e = zeros(T, ndof)
#     zf = zeros(T, N, 2)
#     zg = zeros(T, N, 2)

#     for j in 1:ndof
#         fill!(e, zero(T))
#         e[j] = one(T)
#         apply_linear_elasticity_operator!(ws, e, problem, zf, zg)
#         @views A[:,j] .= ws.r
#     end

#     return sparse(A)
# end

# function solve_linear_elasticity(
#     problem::LinearElasticityProblem{I,T},
#     f::AbstractMatrix{T},
#     g::AbstractMatrix{T};
#     workspace::Union{Nothing,LinearElasticityWorkspace{T}} = nothing,
# ) where {I,T}

#     N = num_nodes(problem.complex.graph)
#     ndof = 2N
#     ws = isnothing(workspace) ? build_linear_elasticity_workspace(problem) : workspace

#     A = assemble_linear_elasticity_matrix(problem, f, g; workspace=ws)

#     U0 = zeros(T, ndof)
#     b = -apply_linear_elasticity_operator!(ws, U0, problem, f, g)

#     U = A \ b
#     return U
# end

# function affine_displacement_field(
#     x::AbstractMatrix{T},
#     A::AbstractMatrix{T},
#     b::AbstractVector{T},
# ) where {T}

#     N = size(x,1)
#     @assert size(x,2) == 2
#     @assert size(A,1) == 2 && size(A,2) == 2
#     @assert length(b) == 2

#     g = zeros(T, N, 2)

#     @inbounds for i in 1:N
#         x1 = x[i,1]
#         x2 = x[i,2]
#         g[i,1] = A[1,1]*x1 + A[1,2]*x2 + b[1]
#         g[i,2] = A[2,1]*x1 + A[2,2]*x2 + b[2]
#     end

#     return g
# end

# function affine_zero_body_force(x::AbstractMatrix{T}) where {T}
#     N = size(x,1)
#     return zeros(T, N, 2)
# end

# # ============================================================
# # Neo-Hookean nonlinear elasticity (PATCHED)
# # ============================================================

# struct NeoHookeanProblem{I<:Integer,T<:Number}
#     complex::MEECComplex
#     gradrec::GradientReconstruction{I,T}
#     λ::T
#     μ::T
# end

# mutable struct NeoHookeanWorkspace{T<:Number}
#     dux_dX::Vector{T}
#     dux_dY::Vector{T}
#     duy_dX::Vector{T}
#     duy_dY::Vector{T}

#     P11::Vector{T}
#     P12::Vector{T}
#     P21::Vector{T}
#     P22::Vector{T}

#     dP11_dX::Vector{T}
#     dP11_dY::Vector{T}
#     dP12_dX::Vector{T}
#     dP12_dY::Vector{T}
#     dP21_dX::Vector{T}
#     dP21_dY::Vector{T}
#     dP22_dX::Vector{T}
#     dP22_dY::Vector{T}

#     r::Vector{T}
#     r_base::Vector{T}

#     J::Matrix{T}
#     du::Vector{T}
#     u_trial::Vector{T}
#     basis::Vector{T}
# end

# function build_neo_hookean_workspace(
#     problem::NeoHookeanProblem{I,T}
# ) where {I,T}
#     N = num_nodes(problem.complex.graph)
#     ndof = 2N
#     return NeoHookeanWorkspace(
#         zeros(T,N), zeros(T,N), zeros(T,N), zeros(T,N),
#         zeros(T,N), zeros(T,N), zeros(T,N), zeros(T,N),
#         zeros(T,N), zeros(T,N), zeros(T,N), zeros(T,N),
#         zeros(T,N), zeros(T,N), zeros(T,N), zeros(T,N),
#         zeros(T,ndof),   # r
#         zeros(T,ndof),   # r_base
#         zeros(T,ndof,ndof),
#         zeros(T,ndof),
#         zeros(T,ndof),
#         zeros(T,ndof),
#     )
# end

# function apply_neo_hookean_operator!(
#     ws::NeoHookeanWorkspace{T},
#     U::AbstractVector{T},
#     problem::NeoHookeanProblem{I,T},
#     f::AbstractMatrix{T},
#     g::AbstractMatrix{T},
# ) where {I,T}

#     complex = problem.complex
#     gradrec = problem.gradrec
#     λ = problem.λ
#     μ = problem.μ

#     N = num_nodes(complex.graph)
#     @assert length(U) == 2N
#     @assert size(f,1) == N && size(f,2) == 2
#     @assert size(g,1) == N && size(g,2) == 2

#     ux = @view U[1:2:2N]
#     uy = @view U[2:2:2N]

#     apply_scalar_gradient!(ws.dux_dX, ws.dux_dY, ux, gradrec)
#     apply_scalar_gradient!(ws.duy_dX, ws.duy_dY, uy, gradrec)

#     @inbounds for i in 1:N
#         F11 = one(T) + ws.dux_dX[i]
#         F12 = ws.dux_dY[i]
#         F21 = ws.duy_dX[i]
#         F22 = one(T) + ws.duy_dY[i]

#         J = F11*F22 - F12*F21

#         if J <= zero(T)
#             J = eps(T)
#         end

#         invJ = one(T) / J

#         Finv11 =  F22 * invJ
#         Finv12 = -F12 * invJ
#         Finv21 = -F21 * invJ
#         Finv22 =  F11 * invJ

#         Fit11 = Finv11
#         Fit12 = Finv21
#         Fit21 = Finv12
#         Fit22 = Finv22

#         logJ = log(J)

#         ws.P11[i] = μ*(F11 - Fit11) + λ*logJ*Fit11
#         ws.P12[i] = μ*(F12 - Fit12) + λ*logJ*Fit12
#         ws.P21[i] = μ*(F21 - Fit21) + λ*logJ*Fit21
#         ws.P22[i] = μ*(F22 - Fit22) + λ*logJ*Fit22
#     end

#     apply_scalar_gradient!(ws.dP11_dX, ws.dP11_dY, ws.P11, gradrec)
#     apply_scalar_gradient!(ws.dP12_dX, ws.dP12_dY, ws.P12, gradrec)
#     apply_scalar_gradient!(ws.dP21_dX, ws.dP21_dY, ws.P21, gradrec)
#     apply_scalar_gradient!(ws.dP22_dX, ws.dP22_dY, ws.P22, gradrec)

#     @inbounds for i in 1:N
#         divP1 = ws.dP11_dX[i] + ws.dP12_dY[i]
#         divP2 = ws.dP21_dX[i] + ws.dP22_dY[i]

#         ws.r[2i-1] = -divP1 - f[i,1]
#         ws.r[2i]   = -divP2 - f[i,2]
#     end

#     @inbounds for i in 1:N
#         if complex.boundary_mask[i]
#             ws.r[2i-1] = ux[i] - g[i,1]
#             ws.r[2i]   = uy[i] - g[i,2]
#         end
#     end

#     return ws.r
# end

# """
# PATCHED finite-difference Jacobian:
# - stores base residual in ws.r_base
# - avoids aliasing bug
# """
# function assemble_neo_hookean_jacobian_fd!(
#     ws::NeoHookeanWorkspace{T},
#     U::AbstractVector{T},
#     problem::NeoHookeanProblem{I,T},
#     f::AbstractMatrix{T},
#     g::AbstractMatrix{T};
#     fd_eps::T = sqrt(eps(T)),
# ) where {I,T}

#     ndof = length(U)

#     apply_neo_hookean_operator!(ws, U, problem, f, g)
#     copyto!(ws.r_base, ws.r)

#     @inbounds for j in 1:ndof
#         copyto!(ws.u_trial, U)
#         ws.u_trial[j] += fd_eps

#         apply_neo_hookean_operator!(ws, ws.u_trial, problem, f, g)
#         @views ws.J[:,j] .= (ws.r .- ws.r_base) ./ fd_eps
#     end

#     # restore base residual to ws.r
#     copyto!(ws.r, ws.r_base)

#     return ws.J
# end

# """
# PATCHED Newton solve:
# - supports good initial guess
# - same residual/Jacobian logic
# - will immediately return if initial guess is already converged
# """
# function solve_neo_hookean(
#     problem::NeoHookeanProblem{I,T},
#     f::AbstractMatrix{T},
#     g::AbstractMatrix{T};
#     u0 = nothing,
#     max_newton::Int = 20,
#     tol::T = T(1e-8),
#     line_search_steps::Int = 10,
#     fd_eps::T = sqrt(eps(T)),
#     verbose::Bool = false,
#     workspace::Union{Nothing,NeoHookeanWorkspace{T}} = nothing,
# ) where {I,T}

#     N = num_nodes(problem.complex.graph)
#     ndof = 2N
#     ws = isnothing(workspace) ? build_neo_hookean_workspace(problem) : workspace

#     U = isnothing(u0) ? zeros(T, ndof) : copy(u0)

#     # enforce Dirichlet dofs strongly in initial guess
#     @inbounds for i in 1:N
#         if problem.complex.boundary_mask[i]
#             U[2i-1] = g[i,1]
#             U[2i]   = g[i,2]
#         end
#     end

#     for k in 1:max_newton
#         apply_neo_hookean_operator!(ws, U, problem, f, g)
#         rnorm = norm(ws.r)

#         if verbose
#             println("Newton iter $k, residual norm = $rnorm")
#         end

#         if rnorm < tol
#             return U
#         end

#         assemble_neo_hookean_jacobian_fd!(ws, U, problem, f, g; fd_eps=fd_eps)

#         ws.du .= -(ws.J \ ws.r)

#         accepted = false
#         α = one(T)

#         for ls in 1:line_search_steps
#             @inbounds for i in 1:ndof
#                 ws.u_trial[i] = U[i] + α * ws.du[i]
#             end

#             @inbounds for i in 1:N
#                 if problem.complex.boundary_mask[i]
#                     ws.u_trial[2i-1] = g[i,1]
#                     ws.u_trial[2i]   = g[i,2]
#                 end
#             end

#             apply_neo_hookean_operator!(ws, ws.u_trial, problem, f, g)
#             trial_norm = norm(ws.r)

#             if verbose
#                 println("  line search $ls: α = $α, residual = $trial_norm")
#             end

#             if trial_norm < rnorm
#                 copyto!(U, ws.u_trial)
#                 accepted = true
#                 break
#             end

#             α /= 2
#         end

#         if !accepted
#             verbose && println("Newton line search failed; exiting.")
#             return U
#         end
#     end

#     return U
# end

# function homogeneous_deformation_displacement(
#     x::AbstractMatrix{T},
#     F0::AbstractMatrix{T},
#     c::AbstractVector{T},
# ) where {T}

#     N = size(x,1)
#     @assert size(x,2) == 2
#     @assert size(F0,1) == 2 && size(F0,2) == 2
#     @assert length(c) == 2

#     g = zeros(T, N, 2)

#     A11 = F0[1,1] - one(T)
#     A12 = F0[1,2]
#     A21 = F0[2,1]
#     A22 = F0[2,2] - one(T)

#     @inbounds for i in 1:N
#         X = x[i,1]
#         Y = x[i,2]
#         g[i,1] = A11*X + A12*Y + c[1]
#         g[i,2] = A21*X + A22*Y + c[2]
#     end

#     return g
# end

# function homogeneous_zero_body_force(x::AbstractMatrix{T}) where {T}
#     return zeros(T, size(x,1), 2)
# end

# end # module MeshFreeExteriorCalculus

module MeshFreeExteriorCalculus

using LinearAlgebra
using SparseArrays

export meshgrid
export EpsilonBallGraph
export MEECComplex
export num_nodes
export num_edges

export GradientReconstruction
export build_gradient_reconstruction
export apply_scalar_gradient!

export NeoHookeanProblem
export NeoHookeanWorkspace
export build_neo_hookean_workspace
export apply_neo_hookean_operator!
export assemble_neo_hookean_jacobian_fd!
export solve_neo_hookean
export solve_neo_hookean_continuation

export homogeneous_deformation_displacement
export homogeneous_zero_body_force
export make_boundary_mask

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
# Graph
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

    # O(N^2) brute force for now
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
    m0::RV
    m1::RV
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
# Gradient reconstruction (column-major nodal fields)
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
# Neo-Hookean nonlinear elasticity (column-major state fields)
# ============================================================

struct NeoHookeanProblem{I<:Integer,T<:Number}
    complex::MEECComplex
    gradrec::GradientReconstruction{I,T}
    λ::T
    μ::T
end

mutable struct NeoHookeanWorkspace{T<:Number}
    # displacement gradients
    dux_dX::Vector{T}
    dux_dY::Vector{T}
    duy_dX::Vector{T}
    duy_dY::Vector{T}

    # first Piola components
    P11::Vector{T}
    P12::Vector{T}
    P21::Vector{T}
    P22::Vector{T}

    # gradients of Piola components
    dP11_dX::Vector{T}
    dP11_dY::Vector{T}
    dP12_dX::Vector{T}
    dP12_dY::Vector{T}
    dP21_dX::Vector{T}
    dP21_dY::Vector{T}
    dP22_dX::Vector{T}
    dP22_dY::Vector{T}

    # residual and jacobian work
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
    reshape(U, 2, N)
end

function apply_neo_hookean_operator!(
    ws::NeoHookeanWorkspace{T},
    U::AbstractVector{T},
    problem::NeoHookeanProblem{I,T},
    f::AbstractMatrix{T},   # (2,N)
    g::AbstractMatrix{T},   # (2,N)
) where {I,T}

    complex = problem.complex
    gradrec = problem.gradrec
    λ = problem.λ
    μ = problem.μ

    N = num_nodes(complex.graph)
    @assert length(U) == 2N
    @assert size(f,1) == 2 && size(f,2) == N
    @assert size(g,1) == 2 && size(g,2) == N

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
        if complex.boundary_mask[i]
            ws.r[2i-1] = Um[1,i] - g[1,i]
            ws.r[2i]   = Um[2,i] - g[2,i]
        end
    end

    return ws.r
end

function assemble_neo_hookean_jacobian_fd!(
    ws::NeoHookeanWorkspace{T},
    U::AbstractVector{T},
    problem::NeoHookeanProblem{I,T},
    f::AbstractMatrix{T},
    g::AbstractMatrix{T};
    fd_eps::T = sqrt(eps(T)),
) where {I,T}

    ndof = length(U)

    apply_neo_hookean_operator!(ws, U, problem, f, g)
    copyto!(ws.r_base, ws.r)

    @inbounds for j in 1:ndof
        copyto!(ws.U_trial, U)
        ws.U_trial[j] += fd_eps

        apply_neo_hookean_operator!(ws, ws.U_trial, problem, f, g)
        @views ws.J[:,j] .= (ws.r .- ws.r_base) ./ fd_eps
    end

    copyto!(ws.r, ws.r_base)
    return ws.J
end

function solve_neo_hookean(
    problem::NeoHookeanProblem{I,T},
    f::AbstractMatrix{T},
    g::AbstractMatrix{T};
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
        if problem.complex.boundary_mask[i]
            Um[1,i] = g[1,i]
            Um[2,i] = g[2,i]
        end
    end

    for k in 1:max_newton
        apply_neo_hookean_operator!(ws, U, problem, f, g)
        rnorm = norm(ws.r)

        verbose && println("Newton iter $k, residual norm = $rnorm")

        if rnorm < tol
            return U
        end

        assemble_neo_hookean_jacobian_fd!(ws, U, problem, f, g; fd_eps=fd_eps)
        ws.du .= -(ws.J \ ws.r)

        accepted = false
        α = one(T)

        for ls in 1:line_search_steps
            @inbounds for i in 1:ndof
                ws.U_trial[i] = U[i] + α * ws.du[i]
            end

            Utrialm = _uview(ws.U_trial, N)
            @inbounds for i in 1:N
                if problem.complex.boundary_mask[i]
                    Utrialm[1,i] = g[1,i]
                    Utrialm[2,i] = g[2,i]
                end
            end

            apply_neo_hookean_operator!(ws, ws.U_trial, problem, f, g)
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
    g::AbstractMatrix{T};
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
    g_step = similar(g)

    for step in 1:nsteps
        α = T(step) / T(nsteps)
        @. g_step = α * g

        if verbose
            println("==================================================")
            println("Continuation step $step / $nsteps, α = $α")
            println("==================================================")
        end

        U = solve_neo_hookean(
            problem,
            f,
            g_step;
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

end # module MeshFreeExteriorCalculusColumnMajor