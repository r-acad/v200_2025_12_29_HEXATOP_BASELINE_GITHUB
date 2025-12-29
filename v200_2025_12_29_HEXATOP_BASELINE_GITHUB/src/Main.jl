# FILE: .\src\Main.jl

using Pkg
using LinearAlgebra
using SparseArrays
using Printf
using Base.Threads
using JSON
using Dates
using Statistics
using CUDA
using YAML

println("\n>>> SCRIPT START: Loading Modules...")
flush(stdout)

const PROJECT_ROOT = abspath(joinpath(@__DIR__, ".."))
const LIMITS_FILE = joinpath(PROJECT_ROOT, "configs", "_machine_limits.jl")

if isfile(LIMITS_FILE)
    include(LIMITS_FILE)
else
    @eval module MachineLimits
        const MAX_GMG_ELEMENTS = 5_000_000
        const MAX_JACOBI_ELEMENTS = 10_000_000
    end
end

module HEXA
    using LinearAlgebra
    using SparseArrays
    using Printf
    using Base.Threads
    using JSON
    using Dates
    using Statistics
    using CUDA
    using YAML
    using ..MachineLimits

    const C_RESET   = "\u001b[0m"
    const C_BOLD    = "\u001b[1m"
    const C_RED     = "\u001b[31m"
    const C_GREEN   = "\u001b[32m"
    const C_YELLOW  = "\u001b[33m"
    const C_BLUE    = "\u001b[34m"
    const C_MAGENTA = "\u001b[35m"
    const C_CYAN    = "\u001b[36m"
    const C_ORANGE  = "\u001b[38;5;208m"

    const PROJECT_ROOT = abspath(joinpath(@__DIR__, ".."))

    # --- Utility Modules ---
    include("Utils/Diagnostics.jl")
    include("Utils/Helpers.jl")
    using .Diagnostics
    using .Helpers

    # --- Core Modules ---
    include("Core/Element.jl")
    include("Core/Boundary.jl")
    include("Core/Stress.jl")
    using .Element
    using .Boundary
    using .Stress

    # --- Mesh Modules ---
    include("Mesh/Mesh.jl")
    include("Mesh/MeshUtilities.jl")
    include("Mesh/MeshPruner.jl")
    include("Mesh/MeshRefiner.jl")
    include("Mesh/MeshShapeProcessing.jl")
    using .Mesh
    using .MeshUtilities
    using .MeshPruner
    using .MeshRefiner
    using .MeshShapeProcessing

    # --- Solver Modules ---
    include("Solvers/CPUSolver.jl")
    include("Solvers/GPUGeometricMultigrid.jl")
    include("Solvers/GPUSolver.jl")
    include("Solvers/DirectSolver.jl")
    include("Solvers/IterativeSolver.jl")
    include("Solvers/Solver.jl")
    
    using .CPUSolver
    using .GPUGeometricMultigrid
    using .GPUSolver
    using .DirectSolver
    using .IterativeSolver
    using .Solver

    # --- IO Modules ---
    include("IO/Configuration.jl")
    include("IO/ExportVTK.jl")
    include("IO/Postprocessing.jl")
    
    # --- Optimization Modules ---
    include("Optimization/GPUExplicitFilter.jl")
    include("Optimization/TopOpt.jl")
    
    using .Configuration
    using .ExportVTK
    using .Postprocessing
    using .GPUExplicitFilter
    using .TopologyOptimization

    # --- Application Logic (Refactored) ---
    include("App/HardwareProfile.jl")
    include("App/SimulationLoop.jl")
    include("App/BatchProcessor.jl")

    using .HardwareProfile
    using .SimulationLoop
    using .BatchProcessor

    function __init__()
        Diagnostics.print_success("HEXA Finite Element Solver initialized")
        Helpers.clear_gpu_memory()
        flush(stdout)
    end
    
    function run(input_file=nothing)
        BatchProcessor.process_batch(input_file, PROJECT_ROOT)
    end

end

function bootstrap()
    println(">>> [BOOTSTRAP] Parsing arguments and launching module...")
    flush(stdout)
    
    config_file = nothing
    if length(ARGS) >= 1
        config_file = ARGS[1]
    end

    HEXA.run(config_file)
end

bootstrap()