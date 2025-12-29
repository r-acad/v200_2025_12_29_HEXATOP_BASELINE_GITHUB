// # FILE: .\src\Optimization\DensityUpdate.jl";


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

include("Filtering.jl")
include("Boundaries.jl")
include("Verification.jl")

function update_density!(density::Vector{Float32}, 
                         l1_stress_norm_field::Vector{Float32}, 
                         protected_elements_mask::BitVector, 
                         E::Float32, 
                         l1_limit_tension::Float32, 
                         l1_limit_compression::Float32,
                         iter::Int, 
                         number_of_iterations::Int, 
                         original_density::Vector{Float32}, 
                         min_density::Float32,  
                         max_density::Float32, 
                         config::Dict, 
                         elements::Matrix{Int};
                         force_no_cull::Bool=false,
                         cutoff_threshold::Float32=0.05f0,
                         specified_radius::Union{Float32, Nothing}=nothing,
                         max_culling_ratio::Float32=0.05f0,
                         update_damping::Float32=0.5f0)  

    nElem = length(density)
    
    # --------------------------------------------------------------------------
    # 1. SAFEGUARD: Check for NaNs
    # --------------------------------------------------------------------------
    if any(isnan, l1_stress_norm_field)
        println("\n" * "\u001b[31m" * "!!!"^20 * "\u001b[0m")
        println("\u001b[31m" * ">>> [SAFEGUARD] CRITICAL: NaNs detected in stress field (Solver Diverged)." * "\u001b[0m")
        println("\u001b[31m" * ">>> [SAFEGUARD] Skipping topology update to prevent mesh corruption." * "\u001b[0m")
        println("\u001b[31m" * "!!!"^20 * "\n" * "\u001b[0m")
        return 0.0f0, 0.0f0, 0.0f0, 0.0, 0, 0.0
    end

    opt_params = config["optimization_parameters"]
    geom_params = config["geometry"]
    
    nElem_x = Int(geom_params["nElem_x_computed"]) 
    nElem_y = Int(geom_params["nElem_y_computed"])
    nElem_z = Int(geom_params["nElem_z_computed"])
    dx = Float32(geom_params["dx_computed"])
    dy = Float32(geom_params["dy_computed"])
    dz = Float32(geom_params["dz_computed"])
    avg_element_size = (dx + dy + dz) / 3.0f0
    
    proposed_density_field = zeros(Float32, nElem)
    
    # --------------------------------------------------------------------------
    # 2. Compute "Proposed" Density (Stress-Based Update)
    # --------------------------------------------------------------------------
    Threads.@threads for e in 1:nElem
        if !protected_elements_mask[e] 
            raw_stress = l1_stress_norm_field[e]
            
            # "Strict" Update Rule:
            limit_e = (raw_stress >= 0) ? l1_limit_tension : l1_limit_compression
            limit_e = max(abs(limit_e), 1.0f-9)
            
            current_l1 = abs(raw_stress)
            target_val = current_l1 / limit_e
            
            # Allow density > 1.0 here so the excess can "diffuse" to neighbors 
            proposed_density_field[e] = max(target_val, min_density)
        else
            # For protected elements (voids/solids), we use their original density 
            # as the source for diffusion.
            proposed_density_field[e] = original_density[e]
        end
    end

    # --------------------------------------------------------------------------
    # 3. Determine Filter Radius
    # --------------------------------------------------------------------------
    R_final = 0.0f0
    if specified_radius !== nothing
        R_final = specified_radius
    else
        target_d_phys = Float32(get(opt_params, "minimum_feature_size_physical", 0.0))
        floor_d_elems = Float32(get(opt_params, "minimum_feature_size_elements", 3.0)) 
        
        floor_d_phys = floor_d_elems * avg_element_size
        d_min_phys = 0.0f0
        if target_d_phys > floor_d_phys
            d_min_phys = target_d_phys
        else
            d_min_phys = floor_d_phys
        end
        
        t = Float32(iter) / Float32(number_of_iterations)
        t = clamp(t, 0.0f0, 1.0f0)

        gamma = Float32(get(opt_params, "radius_decay_exponent", 1.8))
        r_max_mult = Float32(get(opt_params, "radius_max_multiplier", 4.0))
        r_min_mult = Float32(get(opt_params, "radius_min_multiplier", 0.5))
        
        decay_factor = 1.0f0 - (t^gamma)
        r_baseline = (r_max_mult * d_min_phys) * decay_factor + (r_min_mult * d_min_phys)
        R_final = r_baseline
    end
    
    filtered_density_field = proposed_density_field
    filter_time = 0.0
    
    # --------------------------------------------------------------------------
    # 4. Apply Explicit Density Filter (Diffusion)
    # --------------------------------------------------------------------------
    if R_final > 1e-4
        t_start = time()
        
        filtered_density_field = GPUExplicitFilter.apply_explicit_filter!(
            proposed_density_field, 
            nElem_x, nElem_y, nElem_z,
            dx, dy, dz, R_final,
            min_density 
        )
        
        filter_time = time() - t_start
        
        if iter % 10 == 1
            verify_boundary_filtering_detailed(proposed_density_field, filtered_density_field, 
                                               nElem_x, nElem_y, nElem_z)
        end
        
        if any(isnan, filtered_density_field) || any(isinf, filtered_density_field)
            println("\u001b[33m" * ">>> [SAFEGUARD] Filter produced NaNs/Infs. Triggering emergency box filter." * "\u001b[0m")
            
            filtered_density_field = apply_emergency_box_filter(
                proposed_density_field, nElem_x, nElem_y, nElem_z
            )
            
            if any(isnan, filtered_density_field)
                println("\u001b[31m" * ">>> [CRITICAL] Emergency filter also failed. Using unfiltered density." * "\u001b[0m")
                filtered_density_field = proposed_density_field
            end
        end
    end
    
    # Clamp to physical limits
    filtered_density_field = clamp.(filtered_density_field, min_density, max_density)
    
    # --------------------------------------------------------------------------
    # REVISION: Removed Transition Zone Blending
    # --------------------------------------------------------------------------
    # Previously, code here blended the density near boundaries towards 1.0.
    # This caused "sticking" (halos) around Void regions.
    # We now skip this, allowing the optimizer to fully clear material 
    # adjacent to Voids.
    
    effective_cutoff = cutoff_threshold
    update_method = get(opt_params, "density_update_method", "hard")
    
    # --------------------------------------------------------------------------
    # 5. Projection & Culling (Final Design Variable Update)
    # --------------------------------------------------------------------------
    Threads.@threads for e in 1:nElem
        # Only update designable elements
        if !protected_elements_mask[e]
            
            should_cull = filtered_density_field[e] < effective_cutoff
            
            if should_cull
                density[e] = min_density
            else
                if update_method == "hard"
                    density[e] = 1.0f0
                else
                    density[e] = filtered_density_field[e]
                end
            end
        end
    end
    
    # --------------------------------------------------------------------------
    # 6. Strict Boundary Enforcement
    # --------------------------------------------------------------------------
    # Ensure protected elements (Voids and Solids) are exactly their intended value
    Threads.@threads for e in 1:nElem
        if protected_elements_mask[e]
            # This resets voids to min_density (0.001) and solids to their defined value
            density[e] = original_density[e]
        end
    end
    
    delta_rho = mean(abs.(filtered_density_field .- density))
    
    return delta_rho, R_final, effective_cutoff, filter_time, 0, 0.0
end

end