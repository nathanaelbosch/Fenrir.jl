using Fenrir
using Test
using SafeTestsets

@testset "Fenrir.jl" begin
    @safetestset "nll" begin
        include("nll.jl")
    end
end
