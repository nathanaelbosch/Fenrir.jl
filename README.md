# Fenrir: Physics-Enhanced Regression for IVPs

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://nathanaelbosch.github.io/Fenrir.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://nathanaelbosch.github.io/Fenrir.jl/dev)
[![Build Status](https://github.com/nathanaelbosch/Fenrir.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/nathanaelbosch/Fenrir.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/nathanaelbosch/Fenrir.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/nathanaelbosch/Fenrir.jl)

This package exports a single function, `fenrir_nll`, with the following docstring:
```
Compute the "Fenrir" approximate negative log-likelihood (NLL) of the data.

This is a convenience function that

  1. Solves the ODE with a ProbNumDiffEq.EK1 of the specified order and
     with a diffusion as provided by the diffusion_var argument, and

  2. Fits the ODE posterior to the data via Kalman filtering and computes
     the negative log-likelihood on the way.

Returns a tuple (nll::Real, times::Vector{Real}, states::StructVector{Gaussian});
states.μ contains the posterior means, states.Σ the posterior covariances.
```


### Minimal example
Fit data from a FitzHugh-Nagumo model
```julia
using ProbNumDiffEq, Plots, LinearAlgebra, Fenrir

# Define problem:
function f(du, u, p, t)
    a, b, c = p
    du[1] = c*(u[1] - u[1]^3/3 + u[2])
    du[2] = -(1/c)*(u[1] -  a - b*u[2])
end
u0 = [-1.0, 1.0]
tspan = (0.0, 20.0)
p = (0.2, 0.2, 3.0)
prob = ODEProblem(f, u0, tspan, p)

# Generate data:
true_sol = solve(prob, EK1())
times = 1:0.1:20
odedata = true_sol(times).u.μ
scatter(times, ProbNumDiffEq.stack(odedata), markersize=2, markerstrokewidth=0.1,
        color=1, label=["Data" ""])

# With the wrong parameters:
pwrong = (0.1, 0.1, 2.0)
solwrong = solve(remake(prob, p=pwrong), EK1(smooth=false), dense=false);
plot!(solwrong, color=2, label=["Wrong solution" ""])

# Fenrir:
data = (t=times, u=odedata);
σ² = 1e-3
κ² = 1e30
nll, ts, states = fenrir_nll(remake(prob, p=pwrong), data, σ², κ²)

means = ProbNumDiffEq.stack([x.μ for x in states]);
stddevs = ProbNumDiffEq.stack([sqrt.(diag(x.Σ)) for x in states]);

plot!(ts, means, ribbon=2stddevs,
      marker=:o, markersize=1, markerstrokewidth=0.1,
      color=3, fillalpha=0.1, label=["Fenrir interpolation" ""])

println("Negative log-likelihood: $nll")
```

Prints: `Negative log-likelihood: 5849.3096741464615`
