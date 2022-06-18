using Fenrir
using Test

@testset "Fenrir.jl" begin
    @safetestset "nll" begin
        include("nll.jl")
    end
end
