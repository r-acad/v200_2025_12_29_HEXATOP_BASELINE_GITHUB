# FILE: .\src\IO\Postprocessing\MeshTools.jl

using Base.Threads
using LinearAlgebra
using ..Diagnostics

function safe_parse_int(val, default::Int)
    if val === nothing; return default; end
    if isa(val, Number); return Int(val); end
    if isa(val, String)
        clean_val = replace(val, "_" => "")
        return try parse(Int, clean_val) catch; default end
    end
    return default
end

function get_smooth_nodal_densities(density::Vector{Float32}, elements::Matrix{Int}, nNodes::Int)
    node_sums = zeros(Float32, nNodes)
    node_counts = zeros(Int, nNodes)
    nElem = length(density)
    
    @inbounds for e in 1:nElem
        rho = density[e]
        for i in 1:8
            node_idx = elements[e, i]
            if node_idx > 0 && node_idx <= nNodes
                node_sums[node_idx] += rho
                node_counts[node_idx] += 1
            end
        end
    end
    nodal_density = zeros(Float32, nNodes)
    @inbounds for i in 1:nNodes
        if node_counts[i] > 0
            nodal_density[i] = node_sums[i] / Float32(node_counts[i])
        end
    end
    return nodal_density
end

function smooth_grid!(grid::Array{Float32, 3}, passes::Int)
    if passes <= 0; return; end
    nx, ny, nz = size(grid)
    temp_grid = copy(grid)
    
    for _ in 1:passes
        Threads.@threads for k in 2:(nz-1)
            for j in 2:(ny-1)
                for i in 2:(nx-1)
                    sum_neighbors = grid[i-1,j,k] + grid[i+1,j,k] +
                                    grid[i,j-1,k] + grid[i,j+1,k] +
                                    grid[i,j,k-1] + grid[i,j,k+1]
                    temp_grid[i,j,k] = (grid[i,j,k] * 4.0f0 + sum_neighbors) * 0.1f0
                end
            end
        end
        grid[2:end-1, 2:end-1, 2:end-1] .= temp_grid[2:end-1, 2:end-1, 2:end-1]
    end
end

function trilinear_interpolate(vals, xd::Float32, yd::Float32, zd::Float32)
    c00 = vals[1]*(1f0-xd) + vals[2]*xd
    c01 = vals[4]*(1f0-xd) + vals[3]*xd
    c10 = vals[5]*(1f0-xd) + vals[6]*xd
    c11 = vals[8]*(1f0-xd) + vals[7]*xd
    c0 = c00*(1f0-yd) + c01*yd
    c1 = c10*(1f0-yd) + c11*yd
    return c0*(1f0-zd) + c1*zd
end

function decimate_mesh!(vertices::Vector{Tuple{Float64, Float64, Float64}}, 
                        triangles::AbstractVector, 
                        target_triangle_count::Int)
    current_count = length(triangles)
    
    if current_count > 2_000_000 
        Diagnostics.print_warn("Mesh too large for decimation ($current_count tris). Skipping to preserve performance.")
        return triangles
    end

    if target_triangle_count <= 0 || current_count <= target_triangle_count
        return triangles
    end

    Diagnostics.print_info("Decimating mesh: $current_count -> $target_triangle_count triangles...")
    
    mutable_tris = Vector{Vector{Int}}(undef, current_count)
    for i in 1:current_count
        t = triangles[i]
        mutable_tris[i] = [Int(t[1]), Int(t[2]), Int(t[3])]
    end

    max_passes = 15
    for pass in 1:max_passes
        if length(mutable_tris) <= target_triangle_count; break; end

        edges = Vector{Tuple{Float64, Int, Int}}()
        sizehint!(edges, length(mutable_tris) * 3)

        for t in mutable_tris
            v1, v2, v3 = t[1], t[2], t[3]
            d12 = (vertices[v1][1]-vertices[v2][1])^2 + (vertices[v1][2]-vertices[v2][2])^2 + (vertices[v1][3]-vertices[v2][3])^2
            d23 = (vertices[v2][1]-vertices[v3][1])^2 + (vertices[v2][2]-vertices[v3][2])^2 + (vertices[v2][3]-vertices[v3][3])^2
            d31 = (vertices[v3][1]-vertices[v1][1])^2 + (vertices[v3][2]-vertices[v1][2])^2 + (vertices[v3][3]-vertices[v1][3])^2
            push!(edges, (d12, min(v1,v2), max(v1,v2)))
            push!(edges, (d23, min(v2,v3), max(v2,v3)))
            push!(edges, (d31, min(v3,v1), max(v3,v1)))
        end

        sort!(edges, by = x -> x[1])
        
        replacements = collect(1:length(vertices))
        collapsed_nodes = falses(length(vertices))
        n_collapsed = 0
        
        tris_to_remove = length(mutable_tris) - target_triangle_count
        limit_collapses = max(100, tris_to_remove) 

        for (dist, u, v) in edges
            if n_collapsed >= limit_collapses; break; end
            if !collapsed_nodes[u] && !collapsed_nodes[v]
                replacements[v] = u
                mx = (vertices[u][1] + vertices[v][1]) * 0.5
                my = (vertices[u][2] + vertices[v][2]) * 0.5
                mz = (vertices[u][3] + vertices[v][3]) * 0.5
                vertices[u] = (mx, my, mz)
                collapsed_nodes[u] = true 
                collapsed_nodes[v] = true
                n_collapsed += 1
            end
        end

        if n_collapsed == 0; break; end

        new_triangles = Vector{Vector{Int}}()
        sizehint!(new_triangles, length(mutable_tris))

        for t in mutable_tris
            v1 = replacements[t[1]]
            v2 = replacements[t[2]]
            v3 = replacements[t[3]]
            if v1 != v2 && v1 != v3 && v2 != v3
                push!(new_triangles, [v1, v2, v3])
            end
        end
        mutable_tris = new_triangles
    end
    return mutable_tris
end

function laplacian_smooth_mesh!(vertices::Vector{Tuple{Float64, Float64, Float64}}, 
                                triangles::AbstractVector, 
                                iterations::Int=3, lambda::Float64=0.5)
    if iterations <= 0; return; end
    nv = length(vertices)
    new_pos = Vector{Tuple{Float64, Float64, Float64}}(undef, nv)
    neighbor_counts = zeros(Int, nv)
    neighbor_sums_x = zeros(Float64, nv)
    neighbor_sums_y = zeros(Float64, nv)
    neighbor_sums_z = zeros(Float64, nv)

    for _ in 1:iterations
        fill!(neighbor_counts, 0)
        fill!(neighbor_sums_x, 0.0); fill!(neighbor_sums_y, 0.0); fill!(neighbor_sums_z, 0.0)

        for tri in triangles
            i1, i2, i3 = tri[1], tri[2], tri[3]
            if i1 < 1 || i1 > nv || i2 < 1 || i2 > nv || i3 < 1 || i3 > nv; continue; end
            v1 = vertices[i1]; v2 = vertices[i2]; v3 = vertices[i3]
            neighbor_sums_x[i1] += v2[1]; neighbor_sums_y[i1] += v2[2]; neighbor_sums_z[i1] += v2[3]; neighbor_counts[i1] += 1
            neighbor_sums_x[i1] += v3[1]; neighbor_sums_y[i1] += v3[2]; neighbor_sums_z[i1] += v3[3]; neighbor_counts[i1] += 1
            neighbor_sums_x[i2] += v1[1]; neighbor_sums_y[i2] += v1[2]; neighbor_sums_z[i2] += v1[3]; neighbor_counts[i2] += 1
            neighbor_sums_x[i2] += v3[1]; neighbor_sums_y[i2] += v3[2]; neighbor_sums_z[i2] += v3[3]; neighbor_counts[i2] += 1
            neighbor_sums_x[i3] += v1[1]; neighbor_sums_y[i3] += v1[2]; neighbor_sums_z[i3] += v1[3]; neighbor_counts[i3] += 1
            neighbor_sums_x[i3] += v2[1]; neighbor_sums_y[i3] += v2[2]; neighbor_sums_z[i3] += v2[3]; neighbor_counts[i3] += 1
        end
        
        Threads.@threads for i in 1:nv
            cnt = neighbor_counts[i]
            if cnt > 0
                old_x, old_y, old_z = vertices[i]
                avg_x, avg_y, avg_z = neighbor_sums_x[i]/cnt, neighbor_sums_y[i]/cnt, neighbor_sums_z[i]/cnt
                nx = old_x + lambda * (avg_x - old_x)
                ny = old_y + lambda * (avg_y - old_y)
                nz = old_z + lambda * (avg_z - old_z)
                new_pos[i] = (nx, ny, nz)
            else
                new_pos[i] = vertices[i]
            end
        end
        copyto!(vertices, new_pos)
    end
end