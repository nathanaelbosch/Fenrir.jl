using Fenrir
using Test
using SafeTestsets, Aqua, JET

@testset "Fenrir.jl" begin
    @safetestset "fenrir_nll" begin
        include("fenrir_nll.jl")
    end
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(Fenrir, ambiguities=false)
    end
    @testset "Code linting (JET.jl)" begin
        JET.test_package(Fenrir; target_defined_modules=true)
    end
end
