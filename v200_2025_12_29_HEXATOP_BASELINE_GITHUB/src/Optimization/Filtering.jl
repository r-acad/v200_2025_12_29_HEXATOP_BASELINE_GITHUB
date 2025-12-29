# FILE: .\src\Optimization\Filtering.jl

using Base.Threads
using SuiteSparse

mutable struct FilterCache
    is_initialized::Bool
    radius::Float32
    K_filter::SuiteSparse.CHOLMOD.Factor{Float64} 
    FilterCache() = new(false, 0.0f0)
end

const GLOBAL_FILTER_CACHE = FilterCache()

function reset_filter_cache!()
    GLOBAL_FILTER_CACHE.is_initialized = false
end

function apply_emergency_box_filter(density::Vector{Float32}, nx::Int, ny::Int, nz::Int)
    println("    [EMERGENCY FILTER] Applying 3x3x3 box filter (CPU)...")
    
    nElem = length(density)
    filtered = copy(density)
    
    Threads.@threads for k in 2:nz-1
        for j in 2:ny-1
            for i in 2:nx-1
                e = i + (j-1)*nx + (k-1)*nx*ny
                
                if e < 1 || e > nElem; continue; end
                
                sum_rho = 0.0f0
                count = 0
                
                for dk in -1:1, dj in -1:1, di in -1:1
                    neighbor_i = i + di
                    neighbor_j = j + dj
                    neighbor_k = k + dk
                    
                    if neighbor_i >= 1 && neighbor_i <= nx &&
                       neighbor_j >= 1 && neighbor_j <= ny &&
                       neighbor_k >= 1 && neighbor_k <= nz
                        
                        neighbor_idx = neighbor_i + (neighbor_j-1)*nx + (neighbor_k-1)*nx*ny
                        
                        if neighbor_idx >= 1 && neighbor_idx <= nElem
                            sum_rho += density[neighbor_idx]
                            count += 1
                        end
                    end
                end
                
                filtered[e] = (count > 0) ? (sum_rho / count) : density[e]
            end
        end
    end
    
    return filtered
end