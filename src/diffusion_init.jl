function get_initial_diffusion(prob, noisy_ode_data, tsteps, proj=I)
    N = length(tsteps)
    E = length(noisy_ode_data[1])

    integ = init(prob, EK1())
    cache = integ.cache

    @unpack P, PI, A, Q, Ah, Qh = integ.cache
    @unpack measurement, x_filt, x_pred = integ.cache
    @unpack K1, x_tmp2, m_tmp = integ.cache

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
        ProbNumDiffEq.update!(x, x_pred, measurement, R, H, K1, x_tmp2.Σ.mat, m_tmp)

        asdf += z' * (S \ z)
    end
    # @assert tsteps[1] == prob.tspan[1]
    return asdf / N
end