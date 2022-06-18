using Test
using ProbNumDiffEq
using LinearAlgebra
using Fenrir

function f(du, u, p, t)
    a, b, c = p
    du[1] = c * (u[1] - u[1]^3 / 3 + u[2])
    du[2] = -(1 / c) * (u[1] - a - b * u[2])
end
u0 = [-1.0, 1.0]
tspan = (0.0, 20.0)
p = (0.2, 0.2, 3.0)
prob = ODEProblem(f, u0, tspan, p)

# Generate data:
true_sol = solve(prob, EK1())
times = 1:0.1:20
odedata = true_sol(times).u.μ

# With the wrong parameters:
pwrong = (0.1, 0.1, 2.0)
solwrong = solve(remake(prob, p=pwrong), EK1(smooth=false), dense=false);

# Fenrir:
data = (t=times, u=odedata);
σ² = 1e-3
κ² = 1e30
nll, ts, states = fenrir_nll(remake(prob, p=pwrong), data, σ², κ²)

means = ProbNumDiffEq.stack([x.μ for x in states]);
stddevs = ProbNumDiffEq.stack([sqrt.(diag(x.Σ)) for x in states]);
