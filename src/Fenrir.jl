module Fenrir

using LinearAlgebra
using Statistics
using Distributions: logpdf
using ProbNumDiffEq
using ProbNumDiffEq: X_A_Xt, _gaussian_mul!
using UnPack

include("likelihood.jl")
export nll

end
