"""
    fenrir_nll(prob::ODEProblem, data::NamedTuple{(:t, :u)}, observation_noise_var::Real,
        diffusion_var::Union{Real,Vector{<:Real}};
        adaptive=false, dt=false,  proj=I, order=3::Int, tstops=[])

Compute the "Fenrir" approximate negative log-likelihood (NLL) of the data.

This is a convenience function that
1. Solves the ODE with a `ProbNumDiffEq.EK1` of the specified order and with a diffusion
   as provided by the `diffusion_var` argument, and
2. Fits the ODE posterior to the data via Kalman filtering and thereby computes the negative
   log-likelihood on the way.

By default, the solver steps exactly through the time points `data.t`. In addition, you can
provide a step size with `dt` or time stops with `tstops`.
Or, set `adaptive=true` for adaptive step-size selection - use at your own risk!

Returns a tuple `(nll::Real, times::Vector{Real}, states::StructVector{Gaussian})`, where
`states` contains the filtering posterior. Its mean and covariance can be accessed with
`states.μ` and `states.Σ`.

# Arguments
- `prob::ODEProblem`: the initial value problem of interest
- `data::NamedTuple{{(:t, :u)}}`: the data to be fitted
- `observation_noise_var::Real`: the scalar observation noise variance
- `diffusion_var`: the diffusion parameter for the integrated Wiener process prior;
  this plays a similar role as kernel hyperparamers in Gaussian-process regression
- `dt=false`: step size parameter, passed to `OrdinaryDiffEq.init`
- `adaptive=false`: flag to determine if adaptive step size selection should be used;
  use at your own risk!
- `tstops=[]`: additional time stops the algorithm should step through; passed to
  `OrdinaryDiffEq.solve`
- `order::Int=3`: the order of the `ProbNumDiffEq.EK1` solver
- `proj=I`: the matrix which maps the ODE state to the measurements; typically a projection
"""
function fenrir_nll(
    prob::ODEProblem,
    data::NamedTuple{(:t, :u)},
    observation_noise_var::Real,
    diffusion_var::Union{Real,Vector{<:Real}};
    dt=false,
    adaptive::Bool=false,
    tstops=[],
    order=3::Int,
    proj=I,
)

    # Set up the solver with the provided diffusion
    κ² = diffusion_var
    diffmodel = κ² isa Number ? FixedDiffusion(κ², false) : FixedMVDiffusion(κ², false)
    alg = EK1(order=order, diffusionmodel=diffmodel, smooth=false)

    # Create an `integrator` object, and solve the ODE
    integ =
        init(prob, alg, dense=false, tstops=union(data.t, tstops), adaptive=adaptive, dt=dt)
    sol = solve!(integ)

    if sol.retcode != :Success && sol.retcode != :Default
        @error "The PN ODE solver did not succeed!" sol.retcode
        return Inf * one(eltype(integ.p)), sol.t, sol.pu
    end

    # Fit the ODE solution / PN posterior to the provided data; this is the actual Fenrir
    NLL, times, states =
        fit_pnsolution_to_data!(integ, sol, observation_noise_var, data; proj=proj)
    u_probs = project_to_solution_space!(integ.sol.pu, states, integ.cache.SolProj)

    return NLL, times, u_probs
end

function fit_pnsolution_to_data!(
    integ::ProbNumDiffEq.OrdinaryDiffEq.ODEIntegrator{<:ProbNumDiffEq.AbstractEK},
    sol::ProbNumDiffEq.AbstractProbODESolution,
    observation_noise_var::Real,
    data::NamedTuple{(:t, :u)};
    proj=I,
)
    N = length(data.u)
    D = length(integ.u)
    E = length(data.u[1])
    P = length(integ.p)

    R = Diagonal(observation_noise_var .* ones(E))

    NLL = zero(eltype(integ.p))

    @unpack A, Q, x_tmp, x_tmp2, m_tmp = integ.cache
    x_tmp3 = integ.cache.x
    m_tmp = get_lowerdim_measurement_cache(m_tmp, E)

    # x_pred = sol.x_pred # This contains the predicted states of the forward pass
    x_smooth = sol.x_filt # These will be smoothed in the following
    diffusion = sol.diffusions[1] # diffusions are all the same anyways

    H = proj * integ.cache.SolProj
    ZERO_DATA = zeros(E)

    # First update on the last data point
    if sol.t[end] in data.t
        NLL += compute_nll_and_update!(
            x_smooth[end],
            data.u[end],
            H,
            R,
            m_tmp,
            ZERO_DATA,
            integ.cache,
        )
    end

    # Now iterate backwards
    for i in length(x_smooth)-1:-1:1
        dt = sol.t[i+1] - sol.t[i]
        ProbNumDiffEq.make_preconditioners!(integ.cache, dt)
        P, PI = integ.cache.P, integ.cache.PI

        xf = _gaussian_mul!(x_tmp, P, x_smooth[i])
        xs = _gaussian_mul!(x_tmp2, P, x_smooth[i+1])
        # xp = _gaussian_mul!(x_tmp3, P, x_pred[i+1])
        ProbNumDiffEq.smooth!(xf, xs, A, Q, integ.cache, diffusion)
        xs = _gaussian_mul!(x_smooth[i], PI, xf)

        if sol.t[i] in data.t
            data_idx = findfirst(x -> x == sol.t[i], data.t)[1]::Int64
            NLL += compute_nll_and_update!(
                xs,
                data.u[data_idx],
                H,
                R,
                m_tmp,
                ZERO_DATA,
                integ.cache,
            )
        end
    end

    return NLL, sol.t, x_smooth
end

function get_lowerdim_measurement_cache(m_tmp, E)
    _z, _S = m_tmp
    return Gaussian(view(_z, 1:E), PSDMatrix(view(_S.R, :, 1:E)))
end

function measure!(x, H, R, m_tmp)
    z, S = m_tmp
    mul!(z, H, x.μ)
    ProbNumDiffEq.X_A_Xt!(S, x.Σ, H)
    _S = Matrix(S) .+= R
    return Gaussian(z, Symmetric(_S))
end
function update!(
    x_out::SRGaussian,
    x_pred::SRGaussian,
    measurement::Gaussian,
    R::Diagonal,
    H::AbstractMatrix,
    K1_cache::AbstractMatrix,
    K2_cache::AbstractMatrix,
    M_cache::AbstractMatrix,
    C_dxd::AbstractMatrix,
)
    z, S = measurement.μ, measurement.Σ
    m_p, P_p = x_pred.μ, x_pred.Σ
    @assert P_p isa PSDMatrix || P_p isa Matrix
    if (P_p isa PSDMatrix && iszero(P_p.R)) || (P_p isa Matrix && iszero(P_p))
        copy!(x_out, x_pred)
        return x_out
    end

    D = length(m_p)

    # K = P_p * H' / S
    _S = if S isa PSDMatrix
        _matmul!(C_dxd, S.R', S.R)
    else
        copy!(C_dxd, S)
    end

    K = if P_p isa PSDMatrix
        _matmul!(K1_cache, P_p.R, H')
        _matmul!(K2_cache, P_p.R', K1_cache)
    else
        _matmul!(K2_cache, P_p, H')
    end

    S_chol = try
        cholesky!(_S)
    catch e
        if !(e isa PosDefException)
            throw(e)
        end
        @warn "Can't compute the update step with cholesky; using qr instead"
        @assert S isa PSDMatrix
        Cholesky(qr(S.R).R, :U, 0)
    end
    rdiv!(K, S_chol)

    # x_out.μ .= m_p .+ K * (0 .- z)
    x_out.μ .= m_p .- _matmul!(x_out.μ, K, z)

    # M_cache .= I(D) .- mul!(M_cache, K, H)
    _matmul!(M_cache, K, H, -1.0, 0.0)
    @inbounds @simd ivdep for i in 1:D
        M_cache[i, i] += 1
    end

    fast_X_A_Xt!(x_out.Σ, P_p, M_cache)

    out_Sigma_R = [x_out.Σ.R; sqrt.(R) * K']
    x_out.Σ.R .= qr!(out_Sigma_R).R

    return x_out
end
function compute_nll_and_update!(x, u, H, R, m_tmp, ZERO_DATA, cache)
    msmnt = measure!(x, H, R, m_tmp)
    msmnt.μ .-= u
    nll = -logpdf(msmnt, ZERO_DATA)
    # copy!(x, ProbNumDiffEq.update(x, msmnt, H))

    @unpack K1, x_tmp2, m_tmp = cache
    d = length(u)
    # KC, MC, SC = view(K1, :, 1:d), x_tmp2.Σ.mat, view(m_tmp.Σ.mat, 1:d, 1:d)
    xout = cache.x_tmp
    # ProbNumDiffEq.update!(xout, x, msmnt, H, KC, MC, SC)

    @unpack x_tmp2, m_tmp, C_DxD = cache
    C_dxd = view(cache.C_dxd, 1:d, 1:d)
    K1 = view(cache.K1, :, 1:d)
    K2 = view(cache.C_Dxd, :, 1:d)
    update!(xout, x, msmnt, R, H, K1, K2, C_DxD, C_dxd)

    copy!(x, xout)
    return nll
end

function project_to_solution_space!(u_probs, states, projmat)
    for (pu, x) in zip(u_probs, states)
        _gaussian_mul!(pu, projmat, x)
    end
    return u_probs
end
