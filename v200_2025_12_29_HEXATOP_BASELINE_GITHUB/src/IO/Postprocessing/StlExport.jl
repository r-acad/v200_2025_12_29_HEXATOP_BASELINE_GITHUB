# FILE: .\src\IO\Postprocessing\StlExport.jl

using Base.Threads
using ..Diagnostics
using ..Mesh
import MarchingCubes: MC, march

function write_stl_chunked(filename::String, 
                           triangles::AbstractVector, 
                           vertices::Vector{Tuple{Float64, Float64, Float64}})
    
    n_tri = length(triangles)
    
    open(filename, "w") do io
        header_str = "HEXA TopOpt Optimized Binary STL"
        header = zeros(UInt8, 80)
        copyto!(header, 1, codeunits(header_str), 1, min(length(header_str), 80))
        write(io, header)
        write(io, UInt32(n_tri))

        CHUNK_SIZE = 1_000_000 
        buffer = Vector{UInt8}(undef, CHUNK_SIZE * 50)
        
        n_chunks = cld(n_tri, CHUNK_SIZE)
        
        for c in 1:n_chunks
            start_idx = (c - 1) * CHUNK_SIZE + 1
            end_idx = min(c * CHUNK_SIZE, n_tri)
            n_in_chunk = end_idx - start_idx + 1
            
            n_threads = Threads.nthreads()
            batch_size = cld(n_in_chunk, n_threads)
            
            Threads.@threads for t in 1:n_threads
                t_start = start_idx + (t - 1) * batch_size
                t_end = min(start_idx + t * batch_size - 1, end_idx)
                
                if t_start <= t_end
                    for i in t_start:t_end
                        local_i = i - start_idx
                        offset = local_i * 50
                        
                        tri = triangles[i]
                        
                        if tri[1] < 1 || tri[1] > length(vertices) ||
                           tri[2] < 1 || tri[2] > length(vertices) ||
                           tri[3] < 1 || tri[3] > length(vertices)
                            continue
                        end

                        v1 = vertices[tri[1]]
                        v2 = vertices[tri[2]]
                        v3 = vertices[tri[3]]
                        
                        e1x, e1y, e1z = v2[1]-v1[1], v2[2]-v1[2], v2[3]-v1[3]
                        e2x, e2y, e2z = v3[1]-v1[1], v3[2]-v1[2], v3[3]-v1[3]
                        nx, ny, nz = e1y*e2z - e1z*e2y, e1z*e2x - e1x*e2z, e1x*e2y - e1y*e2x
                        mag = sqrt(nx*nx + ny*ny + nz*nz)
                        if mag > 1e-12; nx/=mag; ny/=mag; nz/=mag; else; nx=0.0; ny=0.0; nz=0.0; end
                        
                        ptr = pointer(buffer, offset + 1)
                        p_f32 = reinterpret(Ptr{Float32}, ptr)
                        
                        unsafe_store!(p_f32, Float32(nx), 1)
                        unsafe_store!(p_f32, Float32(ny), 2)
                        unsafe_store!(p_f32, Float32(nz), 3)
                        
                        unsafe_store!(p_f32, Float32(v1[1]), 4)
                        unsafe_store!(p_f32, Float32(v1[2]), 5)
                        unsafe_store!(p_f32, Float32(v1[3]), 6)
                        
                        unsafe_store!(p_f32, Float32(v2[1]), 7)
                        unsafe_store!(p_f32, Float32(v2[2]), 8)
                        unsafe_store!(p_f32, Float32(v2[3]), 9)
                        
                        unsafe_store!(p_f32, Float32(v3[1]), 10)
                        unsafe_store!(p_f32, Float32(v3[2]), 11)
                        unsafe_store!(p_f32, Float32(v3[3]), 12)
                        
                        p_u16 = reinterpret(Ptr{UInt16}, ptr + 48)
                        unsafe_store!(p_u16, UInt16(0), 1)
                    end
                end
            end
            
            bytes_to_write = n_in_chunk * 50
            write(io, view(buffer, 1:bytes_to_write))
        end
    end
end

function export_smooth_watertight_stl(density::Vector{Float32}, geom, threshold::Float32, filename::String; 
                                      subdivision_level::Int=2, smoothing_passes::Int=2, 
                                      mesh_smoothing_iters::Int=3, target_triangle_count::Int=0) 
    
    min_d, max_d = extrema(density)
    if max_d < threshold
        Diagnostics.print_info("Skipping STL: Max density ($max_d) < threshold ($threshold). No surface exists.")
        return
    end

    try
        dir_path = dirname(filename)
        if !isempty(dir_path) && !isdir(dir_path); mkpath(dir_path); end

        NX, NY, NZ = geom.nElem_x, geom.nElem_y, geom.nElem_z
        dx, dy, dz = geom.dx, geom.dy, geom.dz
        
        # --- FIX: Ensure subdivision is at least 1 to prevent DivideError ---
        actual_subdivision = max(1, subdivision_level)
        
        if length(density) > 5_000_000
             actual_subdivision = 1
        end

        nodes_coarse, elements_coarse, _ = Mesh.generate_mesh(NX, NY, NZ; dx=dx, dy=dy, dz=dz)
        nNodes_coarse = size(nodes_coarse, 1)
        if length(density) != size(elements_coarse, 1); return; end
        
        nodal_density_coarse = get_smooth_nodal_densities(density, elements_coarse, nNodes_coarse)
        grid_coarse = reshape(nodal_density_coarse, (NX+1, NY+1, NZ+1))
        smooth_grid!(grid_coarse, smoothing_passes)

        sub_NX, sub_NY, sub_NZ = NX * actual_subdivision, NY * actual_subdivision, NZ * actual_subdivision
        pad = 1 
        fine_dim_x, fine_dim_y, fine_dim_z = sub_NX+1+2*pad, sub_NY+1+2*pad, sub_NZ+1+2*pad
        sub_dx, sub_dy, sub_dz = dx/Float32(actual_subdivision), dy/Float32(actual_subdivision), dz/Float32(actual_subdivision)

        fine_grid = zeros(Float32, fine_dim_x, fine_dim_y, fine_dim_z)
        x_coords = collect(Float32, range(-pad*sub_dx, step=sub_dx, length=fine_dim_x))
        y_coords = collect(Float32, range(-pad*sub_dy, step=sub_dy, length=fine_dim_y))
        z_coords = collect(Float32, range(-pad*sub_dz, step=sub_dz, length=fine_dim_z))

        Threads.@threads for k_f in (1+pad):(fine_dim_z-pad)
            for j_f in (1+pad):(fine_dim_y-pad)
                for i_f in (1+pad):(fine_dim_x-pad)
                    ix, iy, iz = i_f-(1+pad), j_f-(1+pad), k_f-(1+pad)
                    idx_x, idx_y, idx_z = div(ix, actual_subdivision), div(iy, actual_subdivision), div(iz, actual_subdivision)
                    if idx_x >= NX; idx_x = NX - 1; end
                    if idx_y >= NY; idx_y = NY - 1; end
                    if idx_z >= NZ; idx_z = NZ - 1; end
                    c_i, c_j, c_k = idx_x + 1, idx_y + 1, idx_z + 1
                    rem_x, rem_y, rem_z = ix - idx_x*actual_subdivision, iy - idx_y*actual_subdivision, iz - idx_z*actual_subdivision
                    xd, yd, zd = Float32(rem_x)/actual_subdivision, Float32(rem_y)/actual_subdivision, Float32(rem_z)/actual_subdivision
                    vals = (grid_coarse[c_i,c_j,c_k], grid_coarse[c_i+1,c_j,c_k], grid_coarse[c_i+1,c_j+1,c_k], grid_coarse[c_i,c_j+1,c_k],
                            grid_coarse[c_i,c_j,c_k+1], grid_coarse[c_i+1,c_j,c_k+1], grid_coarse[c_i+1,c_j+1,c_k+1], grid_coarse[c_i,c_j+1,c_k+1])
                    fine_grid[i_f, j_f, k_f] = trilinear_interpolate(vals, xd, yd, zd)
                end
            end
        end

        mc_struct = MC(fine_grid, Int; normal_sign=1, x=x_coords, y=y_coords, z=z_coords)
        march(mc_struct, threshold)
        
        if length(mc_struct.vertices) == 0
            Diagnostics.print_warn("STL generation produced 0 vertices (Empty).")
            return
        end

        final_triangles = collect(mc_struct.triangles) 
        final_vertices = [(Float64(v[1]), Float64(v[2]), Float64(v[3])) for v in mc_struct.vertices]

        if mesh_smoothing_iters > 0
            try
                verts_tuple = copy(final_vertices)
                laplacian_smooth_mesh!(verts_tuple, final_triangles, mesh_smoothing_iters, 0.5)
                final_vertices = verts_tuple
            catch e
                Diagnostics.print_warn("Mesh smoothing failed: $e")
            end
        end

        if target_triangle_count > 0 && length(final_triangles) > target_triangle_count
             try
                 final_triangles = decimate_mesh!(final_vertices, final_triangles, target_triangle_count)
             catch e
                 Diagnostics.print_error("Mesh decimation failed ($e). Exporting un-decimated mesh.")
             end
        end

        write_stl_chunked(filename, final_triangles, final_vertices)

    catch e
        Diagnostics.print_error("STL Export crashed: $e")
        Diagnostics.write_crash_log("crash_log.txt", "STL_EXPORT", e, stacktrace(catch_backtrace()), 0, Dict(), Float32[])
    end
end