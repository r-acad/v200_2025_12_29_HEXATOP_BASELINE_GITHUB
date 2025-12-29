# FILE: .\src\App\HardwareProfile.jl
module HardwareProfile

using CUDA
using ..Diagnostics
using ..MachineLimits

export apply_hardware_profile!

function apply_hardware_profile!(config::Dict)
    gpu_type = get(config, "gpu_profile", "RTX")
    if gpu_type == "AUTO" && CUDA.functional()
        dev_name = CUDA.name(CUDA.device())
        if occursin("V100", dev_name); gpu_type = "V100"; end
        if occursin("A100", dev_name) || occursin("H100", dev_name); gpu_type = "H200"; end
        Diagnostics.print_info("Auto-Detected GPU: $dev_name -> Profile: $gpu_type")
    else
        Diagnostics.print_info("Using Configured Profile: $gpu_type")
    end
    
    mesh_conf = get(config, "mesh_settings", Dict())
    solver = get(config, "solver_parameters", Dict())
    
    config["force_float64"] = false
    if uppercase(gpu_type) in ["H", "H200", "H100", "A100"]
        Diagnostics.print_substep("High-Performance Data Center GPU (H/A-Series). Precision: Float64.")
        solver["tolerance"] = get(solver, "tolerance", 1.0e-12)
        solver["diagonal_shift_factor"] = 1.0e-10
        solver["solver_type"] = "gpu"
        config["force_float64"] = true
    elseif uppercase(gpu_type) == "V100"
        Diagnostics.print_substep("Legacy Data Center GPU (Tesla V100). Precision: Float64.")
        solver["tolerance"] = get(solver, "tolerance", 1.0e-10)
        solver["diagonal_shift_factor"] = 1.0e-9
        solver["solver_type"] = "gpu"
        config["force_float64"] = true
    else 
        Diagnostics.print_substep("Consumer/Workstation GPU (RTX-Series). Precision: Float32.")
        solver["tolerance"] = get(solver, "tolerance", 1.0e-6)
        solver["solver_type"] = "gpu"
        config["force_float64"] = false
    end
    config["mesh_settings"] = mesh_conf
    config["solver_parameters"] = solver
    config["hardware_profile_applied"] = gpu_type
    
    config["machine_limits"] = Dict(
        "MAX_GMG_ELEMENTS" => MachineLimits.MAX_GMG_ELEMENTS,
        "MAX_JACOBI_ELEMENTS" => MachineLimits.MAX_JACOBI_ELEMENTS
    )
end

end