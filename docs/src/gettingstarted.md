# Computing Approximate Likelihoods with Probabilistic Numerics and Fenrir.jl

```@example 1
using LinearAlgebra
using OrdinaryDiffEq, ProbNumDiffEq, Plots
using Fenrir
stack(x) = copy(reduce(hcat, x)') # convenient
nothing # hide
```

## The problem statement as math
Let's assume we have an initial value problem (IVP)
```math
\begin{aligned}
\dot{y} &= f_\theta(y, t), \qquad y(t_0) = y_0,
\end{aligned}
```
which we observe through a set ``\mathcal{D} = \{u(t_n)\}_{n=1}^N`` of noisy data points
```math
\begin{aligned}
u(t_n) = H y(t_n) + v_n, \qquad v_n \sim \mathcal{N}(0, R).
\end{aligned}
```
The question of interest is: How can we compute the marginal likelihood ``p(\mathcal{D} \mid \theta)``?
Short answer: We can't. It's intractable, because exactly computing the true IVP solution ``y(t)`` is intractable.
What we can do however is compute an approximate marginal likelihood.
This is what Fenrir.jl provides.
For details, check out the [paper](https://arxiv.org/abs/2202.01287).

## The setup, in code
Let's assume that the true underlying dynamics are given by a FitzHugh-Nagumo model
```@example 1
function f(du, u, p, t)
    a, b, c = p
    du[1] = c*(u[1] - u[1]^3/3 + u[2])
    du[2] = -(1/c)*(u[1] -  a - b*u[2])
end
u0 = [-1.0, 1.0]
tspan = (0.0, 20.0)
p = (0.2, 0.2, 3.0)
true_prob = ODEProblem(f, u0, tspan, p)
```
from which we generate some artificial noisy data
```@example 1
true_sol = solve(true_prob, Vern9(), abstol=1e-10, reltol=1e-10)

times = 1:0.5:20
observation_noise_var = 1e-1
odedata = [true_sol(t) .+ sqrt(observation_noise_var) * randn(length(u0)) for t in times]

plot(true_sol, color=:black, linestyle=:dash, label=["True Solution" ""])
scatter!(times, stack(odedata), markersize=2, markerstrokewidth=0.1, color=1, label=["Noisy Data" ""])
```

## Computing the negative log-likelihood
To evaluate the likelihood given a parameter estimate ``\theta_\text{est}``,
we just need to call `fenrir_nll`:
```@example 1
p_est = (0.1, 0.1, 2.0)
prob = remake(true_prob, p=p_est)
data = (t=times, u=odedata)
κ² = 1e10
nll, _, _ = fenrir_nll(prob, data, observation_noise_var, κ²; dt=1e-1)
nll
```
Voilà! This is the marginal negative log-likelihood!

You can use it as any other NLL: Optimize it to compute maximum-likelihood estimates or MAPs, or plug it into MCMC to sample from the posterior.
In [our paper](https://arxiv.org/abs/2202.01287) we compute MLEs by pairing Fenrir with [Optimization.jl](http://optimization.sciml.ai/stable/) and [ForwardDiff.jl](https://juliadiff.org/ForwardDiff.jl/stable/).
