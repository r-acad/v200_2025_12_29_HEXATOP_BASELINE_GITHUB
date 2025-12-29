// # FILE: .\src\IO\Postprocessing\WebExport.jl";


using JSON
using Base.Threads
using ..MeshUtilities
using ..Diagnostics

function export_binary_for_web(filename::String, 
                               nodes::Matrix{Float32}, 
                               elements::Matrix{Int}, 
                               density::Vector{Float32}, 
                               l1_stress::Vector{Float32}, 
                               principal_field::Matrix{Float32}, 
                               geom, 
                               threshold::Float32, 
                               iter::Int, 
                               current_radius::Float32, 
                               integral_error::Float32, 
                               config::Dict; 
                               max_export_cells::Int=0) 
    
    all_active_indices = findall(x -> x >= threshold, density)
    n_active = length(all_active_indices)
    if n_active == 0; return; end

    if max_export_cells > 0 && n_active > max_export_cells
        step_val = n_active / max_export_cells
        indices_to_export = Int[]
        sizehint!(indices_to_export, max_export_cells)
        curr_float_idx = 1.0
        while curr_float_idx <= n_active
            idx_int = floor(Int, curr_float_idx)
            if idx_int <= n_active
                push!(indices_to_export, all_active_indices[idx_int])
            end
            curr_float_idx += step_val
        end
        valid_indices = indices_to_export
    else
        valid_indices = all_active_indices
    end

    count = length(valid_indices)

    # 1. Copy the full configuration for metadata
    meta = deepcopy(config)
    meta["iteration"] = iter
    meta["radius"] = current_radius
    meta["threshold"] = threshold
    
    # 2. Add restart/stats data
    meta["restart_data"] = Dict(
        "integral_error" => integral_error,
        "n_nodes" => size(nodes, 1),
        "n_elements" => size(elements, 1)
    )

    if !haskey(meta, "mesh_settings"); meta["mesh_settings"] = Dict(); end
    meta["mesh_settings"]["initial_ground_mesh_size"] = size(elements, 1)

    # 3. Process geometry specifically to help the viewer
    # The viewer looks for an "action" field ("add" or "remove") to color shapes.
    if haskey(meta, "geometry") && isa(meta["geometry"], Dict)
        for (key, shape) in meta["geometry"]
            if isa(shape, Dict) && haskey(shape, "type")
                if !haskey(shape, "action")
                    
                    stiffness_val = nothing
                    
                    # Check legacy root key
                    if haskey(shape, "stiffness_ratio")
                        stiffness_val = Float32(shape["stiffness_ratio"])
                    end
                    
                    # Check new 'properties' sub-dictionary (this overrides root)
                    if haskey(shape, "properties") && isa(shape["properties"], Dict)
                        props = shape["properties"]
                        if haskey(props, "stiffness_ratio")
                            stiffness_val = Float32(props["stiffness_ratio"])
                        end
                    end
                    
                    # Infer action based on stiffness
                    if stiffness_val !== nothing
                        shape["action"] = stiffness_val > 0.0f0 ? "add" : "remove"
                    else
                        # Default fallback if no stiffness defined
                        shape["action"] = "add"
                    end
                end
            end
        end
    end

    meta["loads"] = get(config, "external_forces", [])
    meta["bcs"] = get(config, "boundary_conditions", [])
    meta["settings"] = get(config, "optimization_parameters", Dict())
    
    json_str = JSON.json(meta)
    json_bytes = Vector{UInt8}(json_str)
    json_len = UInt32(length(json_bytes))

    try
        open(filename, "w") do io
            write(io, 0x48455841) # Magic "HEXA"
            write(io, UInt32(2))  # Version
            write(io, Int32(iter))
            write(io, Float32(current_radius))
            write(io, Float32(threshold))
            write(io, UInt32(count))
            write(io, Float32(geom.dx))
            write(io, Float32(geom.dy))
            write(io, Float32(geom.dz))

            centroids = zeros(Float32, count * 3)
            densities = zeros(Float32, count)
            signed_l1 = zeros(Float32, count)

            Threads.@threads for i in 1:count
                idx = valid_indices[i]
                c = MeshUtilities.element_centroid(idx, nodes, elements)
                centroids[3*(i-1)+1] = c[1]
                centroids[3*(i-1)+2] = c[2]
                centroids[3*(i-1)+3] = c[3]

                densities[i] = density[idx]

                # Store Signed L1 Stress for visualization
                signed_l1[i] = l1_stress[idx]
            end

            write(io, centroids)
            write(io, densities)
            write(io, signed_l1)
            
            write(io, json_len)
            write(io, json_bytes)
        end
    catch e
        Diagnostics.print_error("[Binary Export] Failed to write file: $e")
    end
end