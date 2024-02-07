module Fenrir

using LinearAlgebra
using Statistics
using Distributions: logpdf
using ProbNumDiffEq
import ProbNumDiffEq as PNDE
using SimpleUnPack

include("likelihood.jl")
export fenrir_nll

function __init__()
    # Deprecation warning: This package will not be maintained in the future.
    @warn("""\n
        # Deprecation warning: This package will not be maintained in the future.
        The `fenrir_nll` function implemented in this package is now implemented in
        ProbNumDiffEq.jl, together with other data likelihood functions. So, instead of
        using Fenrir.jl, use ProbNumDiffEq.jl.
    """)
end

end
