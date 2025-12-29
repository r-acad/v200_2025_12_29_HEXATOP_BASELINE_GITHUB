# FILE: .\src\Optimization\Boundaries.jl

using Base.Threads
using LinearAlgebra

function create_transition_zone(protected_mask::BitVector, nx::Int, ny::Int, nz::Int, depth::Int=3)
    nElem = length(protected_mask)
    transition_zone = falses(nElem)
    
    for k in 1:nz, j in 1:ny, i in 1:nx
        e = i + (j-1)*nx + (k-1)*nx*ny
        if e < 1 || e > nElem; continue; end
        
        if protected_mask[e]; continue; end
        
        # Check if within 'depth' layers of a protected element
        found_protected = false
        for dk in -depth:depth, dj in -depth:depth, di in -depth:depth
            ni = i + di
            nj = j + dj
            nk = k + dk
            
            if ni >= 1 && ni <= nx && nj >= 1 && nj <= ny && nk >= 1 && nk <= nz
                neighbor_idx = ni + (nj-1)*nx + (nk-1)*nx*ny
                if neighbor_idx >= 1 && neighbor_idx <= nElem && protected_mask[neighbor_idx]
                    found_protected = true
                    break
                end
            end
        end
        
        if found_protected
            transition_zone[e] = true
        end
    end
    
    return transition_zone
end

function blend_transition_zone!(density::Vector{Float32}, 
                                filtered_density::Vector{Float32},
                                protected_mask::BitVector,
                                transition_zone::BitVector,
                                original_density::Vector{Float32},
                                nx::Int, ny::Int, nz::Int,
                                blend_depth::Int=3)
    
    nElem = length(density)
    
    Threads.@threads for k in 1:nz
        for j in 1:ny
            for i in 1:nx
                e = i + (j-1)*nx + (k-1)*nx*ny
                if e < 1 || e > nElem; continue; end
                
                if !transition_zone[e]; continue; end
                
                min_dist = blend_depth + 1.0
                
                for dk in -blend_depth:blend_depth, dj in -blend_depth:blend_depth, di in -blend_depth:blend_depth
                    ni = i + di
                    nj = j + dj
                    nk = k + dk
                    
                    if ni >= 1 && ni <= nx && nj >= 1 && nj <= ny && nk >= 1 && nk <= nz
                        neighbor_idx = ni + (nj-1)*nx + (nk-1)*nx*ny
                        if neighbor_idx >= 1 && neighbor_idx <= nElem && protected_mask[neighbor_idx]
                            dist = sqrt(Float32(di^2 + dj^2 + dk^2))
                            min_dist = min(min_dist, dist)
                        end
                    end
                end
                
                alpha = clamp(min_dist / blend_depth, 0.0f0, 1.0f0)
                smooth_alpha = alpha * alpha * (3.0f0 - 2.0f0 * alpha)
                density[e] = (1.0f0 - smooth_alpha) * original_density[e] + smooth_alpha * filtered_density[e]
            end
        end
    end
end