# FILE: .\src\Optimization\TopOpt.jl

module TopologyOptimization 

using LinearAlgebra
using SparseArrays
using Printf  
using Statistics 
using SuiteSparse 
using CUDA
using Base.Threads

using ..Element
using ..Mesh
using ..GPUExplicitFilter
using ..Helpers

export update_density!, reset_filter_cache!

# Include sub-modules by concern
include("Filtering.jl")
include("Boundaries.jl")
include("Verification.jl")
include("DensityUpdate.jl")

end