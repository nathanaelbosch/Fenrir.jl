using Fenrir
using Test
using SafeTestsets, Aqua, JET

const GROUP = get(ENV, "GROUP", "All")

@testset "Fenrir.jl" begin
    if GROUP == "All" || GROUP == "Downstream"
        @safetestset "fenrir_nll" begin
            include("fenrir_nll.jl")
        end
    end

    if GROUP == "All"
        @testset "Code quality (Aqua.jl)" begin
            Aqua.test_all(Fenrir, ambiguities=false)
        end
        if VERSION >= v"1.9"
            @testset "Code linting (JET.jl)" begin
                JET.test_package(Fenrir; target_defined_modules=true)
            end
        end
    end
end
