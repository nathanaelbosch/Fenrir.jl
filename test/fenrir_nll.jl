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

proj = [1 0]

# Generate data:
true_sol = solve(prob, EK1())
times = 0.1:0.1:20
odedata = [proj * true_sol(t).μ for t in times]

# With the wrong parameters:
pwrong = (0.1, 0.1, 2.0)
# solwrong = solve(remake(prob, p=pwrong), EK1(smooth=false), dense=false);

# Fenrir:
data = (t=times, u=odedata);
σ² = 1e-3

@testset "Scalar diffusion" begin
    κ² = 1e30
    nll, ts, states = fenrir_nll(remake(prob, p=pwrong), data, σ², κ², proj=proj)
    @test nll isa Number
    @test !isinf(nll)
    @test ts isa Vector{<:Number}
    @test states isa StructVector{<:Gaussian}

    means = ProbNumDiffEq.stack([x.μ for x in states])
    stddevs = ProbNumDiffEq.stack([sqrt.(diag(x.Σ)) for x in states])
end

@testset "Vector-valued diffusion" begin
    κ² = 1e30 * ones(length(u0))
    nll, ts, states = fenrir_nll(remake(prob, p=pwrong), data, σ², κ², proj=proj)
    @test nll isa Number
    @test !isinf(nll)
    @test ts isa Vector{<:Number}
    @test states isa StructVector{<:Gaussian}

    means = ProbNumDiffEq.stack([x.μ for x in states])
    stddevs = ProbNumDiffEq.stack([sqrt.(diag(x.Σ)) for x in states])
end

@testset "Custom Algorithm Interface" begin
    @testset "The default case with fixed-diffusion EK1" begin
        κ² = 1e30
        alg = EK1(order=3, smooth=true, diffusionmodel=FixedDiffusion(κ², false))
        nll, ts, states = fenrir_nll(remake(prob, p=pwrong), data, alg, σ², proj=proj)
        @test nll isa Number
        @test !isinf(nll)
        @test ts isa Vector{<:Number}
        @test states isa StructVector{<:Gaussian}
    end
    @testset "EK0" begin
        κ² = 1e30
        alg = EK0(order=3, smooth=true, diffusionmodel=FixedDiffusion(κ², false))
        nll, ts, states = fenrir_nll(remake(prob, p=pwrong), data, alg, σ², proj=proj)
        @test nll isa Number
        @test !isinf(nll)
        @test ts isa Vector{<:Number}
        @test states isa StructVector{<:Gaussian}
    end
    @testset "smooth=false" begin
        κ² = 1e30
        alg = EK1(order=3, smooth=false, diffusionmodel=FixedDiffusion(κ², false))
        @test_throws ArgumentError fenrir_nll(remake(prob, p=pwrong), data, alg, σ², proj=proj)
    end
    @testset "DynamicDiffusion" begin
        alg = EK1(order=3, smooth=true, diffusionmodel=DynamicDiffusion())
        @test_warn "not recommended" fenrir_nll(remake(prob, p=pwrong), data, alg, σ², proj=proj)
        nll, ts, states = fenrir_nll(remake(prob, p=pwrong), data, alg, σ², proj=proj)
        @test nll isa Number
        @test !isinf(nll)
        @test ts isa Vector{<:Number}
        @test states isa StructVector{<:Gaussian}
    end
    @testset "calibrate=true" begin
        κ² = 1e30
        alg = EK1(order=3, smooth=true, diffusionmodel=FixedDiffusion(κ², true))
        @test_warn "not recommended" fenrir_nll(remake(prob, p=pwrong), data, alg, σ², proj=proj)
        nll, ts, states = fenrir_nll(remake(prob, p=pwrong), data, alg, σ², proj=proj)
        @test nll isa Number
        @test !isinf(nll)
        @test ts isa Vector{<:Number}
        @test states isa StructVector{<:Gaussian}
    end
end
