# Fenrir: Physics-Enhanced Regression for IVPs

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://nathanaelbosch.github.io/Fenrir.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://nathanaelbosch.github.io/Fenrir.jl/dev)
[![Build Status](https://github.com/nathanaelbosch/Fenrir.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/nathanaelbosch/Fenrir.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/nathanaelbosch/Fenrir.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/nathanaelbosch/Fenrir.jl)

This package exports a single function, `fenrir_nll`:
```
    fenrir_nll(prob::ODEProblem, data::NamedTuple{(:t, :u)}, observation_noise_var::Real,
        diffusion_var::Real; adaptive=false, dt=false,  proj=I, order=3::Int, tstops=[])

Compute the "Fenrir" approximate negative log-likelihood (NLL) of the data.
```
To see the full docstring, check out the
[documentation](https://nathanaelbosch.github.io/Fenrir.jl/stable).


### Minimal example: Parameter likelihood on FitzHugh-Nagumo
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
odedata = [u + 0.1*randn(size(u)) for u in true_sol(times).u.μ]
scatter(times, ProbNumDiffEq.stack(odedata), markersize=2, markerstrokewidth=0.1,
        color=1, label=["Data" ""])

# With the wrong parameters:
pwrong = (0.1, 0.1, 2.0)
solwrong = solve(remake(prob, p=pwrong), EK1(smooth=false), dense=false);
plot!(solwrong, color=2, label=["Wrong solution" ""])

# Fenrir:
data = (t=times, u=odedata);
σ² = 1e-1
κ² = 1e30
nll, ts, states = fenrir_nll(remake(prob, p=pwrong), data, σ², κ²)

means = ProbNumDiffEq.stack([x.μ for x in states]);
stddevs = ProbNumDiffEq.stack([sqrt.(diag(x.Σ)) for x in states]);

plot!(ts, means, ribbon=2stddevs,
      marker=:o, markersize=1, markerstrokewidth=0.1,
      color=3, fillalpha=0.1, label=["Fenrir interpolation" ""])

println("Negative log-likelihood: $nll")
```
![README Demo](./docs/src/readmedemo.svg?raw=true "README Demo")

Prints: `Negative log-likelihood: 5849.3096741464615`

You can use this NLL it as any other NLL:
Optimize it to compute maximum-likelihood estimates or MAPs,
or plug it into MCMC to sample from the posterior.
In [our paper](https://arxiv.org/abs/2202.01287) we compute MLEs by pairing Fenrir with [Optimization.jl](http://optimization.sciml.ai/stable/) and [ForwardDiff.jl](https://juliadiff.org/ForwardDiff.jl/stable/).



## Reference
This method has been developed in the paper "Fenrir: Physics-Enhanced Regression for Initial Value Problems" by Tronarp et al, published at ICML 2022 ([link](https://proceedings.mlr.press/v162/tronarp22a.html)).
```

@InProceedings{pmlr-v162-tronarp22a,
  title = 	 {Fenrir: Physics-Enhanced Regression for Initial Value Problems},
  author =       {Tronarp, Filip and Bosch, Nathanael and Hennig, Philipp},
  booktitle = 	 {Proceedings of the 39th International Conference on Machine Learning},
  pages = 	 {21776--21794},
  year = 	 {2022},
  editor = 	 {Chaudhuri, Kamalika and Jegelka, Stefanie and Song, Le and Szepesvari, Csaba and Niu, Gang and Sabato, Sivan},
  volume = 	 {162},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {17--23 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v162/tronarp22a/tronarp22a.pdf},
  url = 	 {https://proceedings.mlr.press/v162/tronarp22a.html}
}
```
