module Fenrir

using LinearAlgebra
using Statistics
using Distributions: logpdf
using ProbNumDiffEq
using ProbNumDiffEq: X_A_Xt, _gaussian_mul!, SRGaussian, _matmul!, fast_X_A_Xt!, triangularize!
using UnPack

include("likelihood.jl")
export fenrir_nll

end
