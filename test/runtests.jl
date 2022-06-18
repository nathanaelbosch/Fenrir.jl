using Fenrir
using Test
using SafeTestsets

@testset "Fenrir.jl" begin
    @safetestset "fenrir_nll" begin
        include("fenrir_nll.jl")
    end
end
