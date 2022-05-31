"""
Compute the "Fenrir" approximate negative log-likelihood (NLL) of the data.

This is a convenience function that
1. Solves the ODE with a `ProbNumDiffEq.EK1` of the specified order and with a diffusion
   as provided by the `diffusion_var` argument, and
2. Fits the ODE posterior to the data via Kalman filtering and computes the negative
   log-likelihood on the way.

Returns a tuple `(nll::Real, times::Vector{Real}, states::StructVector{Gaussian})`;
`states.μ` contains the posterior means, `states.Σ` the posterior covariances.
"""
function nll(
    ode_problem::ODEProblem,
    data::NamedTuple{(:t, :u)},
    observation_noise_var::Real,
    diffusion_var::Real;
    adaptive=false,
    dt=false,
    proj=I,
    order=3::Int,
    tstops=[],
)
    # Set up the solver with the provided diffusion
    κ² = diffusion_var
    diffmodel = κ² isa Number ? FixedDiffusion(κ², false) : FixedMVDiffusion(κ², false)
    alg = EK1(order=order, diffusionmodel=diffmodel, smooth=false)

    # Create an `integrator` object, and solve the ODE
    integ = init(
        ode_problem,
        alg,
        dense=false,
        tstops=union(data.t, tstops),
        adaptive=adaptive,
        dt=dt,
    )
    sol = solve!(integ)

    if sol.retcode != :Success
        @error "The PN ODE solver did not succeed!"
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
function compute_nll_and_update!(x, u, H, R, m_tmp, ZERO_DATA, cache)
    msmnt = measure!(x, H, R, m_tmp)
    msmnt.μ .-= u
    nll = -logpdf(msmnt, ZERO_DATA)
    # copy!(x, ProbNumDiffEq.update(x, msmnt, H))

    @unpack K1, K2, x_tmp2, m_tmp = cache
    d = length(u)
    # KC, MC, SC = view(K1, :, 1:d), x_tmp2.Σ.mat, view(m_tmp.Σ.mat, 1:d, 1:d)
    xout = cache.x_tmp
    # ProbNumDiffEq.update!(xout, x, msmnt, H, KC, MC, SC)

    @unpack K1, K2, x_tmp2, m_tmp, C_DxD = cache
    ProbNumDiffEq.update!(xout, x, msmnt, H, K1, C_DxD, cache.C_dxd)

    copy!(x, xout)
    return nll
end

function project_to_solution_space!(u_probs, states, projmat)
    for (pu, x) in zip(u_probs, states)
        _gaussian_mul!(pu, projmat, x)
    end
    return u_probs
end

function get_initial_diff(prob, noisy_ode_data, tsteps, proj=I)
    N = length(tsteps)
    E = length(noisy_ode_data[1])

    integ = init(prob, EK1())
    cache = integ.cache

    @unpack P, PI, A, Q, Ah, Qh = integ.cache
    @unpack measurement, x_filt, x_pred = integ.cache
    @unpack K1, K2, x_tmp2, m_tmp = integ.cache

    m_tmp = get_lowerdim_measurement_cache(m_tmp, E)
    measurement = get_lowerdim_measurement_cache(measurement, E)
    K1 = view(K1, :, 1:E)

    H = proj * cache.SolProj
    x = copy(cache.x)
    x.μ .= 0
    D, _ = size(x.Σ)
    σ = 1e2
    x.Σ.squareroot .= σ * I(D)
    x.Σ.mat .= σ^2 * I(D)

    t0 = prob.tspan[1]
    asdf = zero(eltype(x.μ))
    for i in 1:N
        dt = i == 1 ? tsteps[i] - t0 : tsteps[i] - tsteps[i-1]

        if dt > 0
            ProbNumDiffEq.make_preconditioners!(cache, dt) # updates P and PI
            @. Ah .= PI.diag .* A .* P.diag'
            ProbNumDiffEq.X_A_Xt!(Qh, Q, PI)

            ProbNumDiffEq.predict!(x_pred, x, Ah, Qh, cache.C1)
        else
            copy!(x_pred, x)
        end

        z, S = measurement
        mul!(z, H, x_pred.μ)
        z .-= noisy_ode_data[i]
        ProbNumDiffEq.X_A_Xt!(S, x_pred.Σ, H)
        ProbNumDiffEq.update!(x, x_pred, measurement, H, K1, x_tmp2.Σ.mat, m_tmp)

        asdf += z' * (S \ z)
    end
    # @assert tsteps[1] == prob.tspan[1]
    return asdf / N
end
