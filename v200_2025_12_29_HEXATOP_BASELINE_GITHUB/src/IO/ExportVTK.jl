# FILE: .\src\IO\ExportVTK.jl
module ExportVTK 

using Printf 
using Base64 

export export_solution_vti, export_solution_legacy, export_solution

"""
    export_solution_vti(...)

Writes the simulation results to a VTK XML Image Data file (.vti).
FIXED: Removed newline before binary marker to prevent byte-alignment errors.
"""
function export_solution_vti(dims::Tuple{Int,Int,Int}, 
                             spacing::Tuple{Float32,Float32,Float32}, 
                             origin::Tuple{Float32,Float32,Float32},
                             density::Vector{Float32}, 
                             l1_stress::Vector{Float32},
                             von_mises::Vector{Float32},
                             nodal_displacement::Matrix{Float32}, 
                             principal_vals::Matrix{Float32},
                             principal_max_dirs::Matrix{Float32},
                             principal_min_dirs::Matrix{Float32},
                             config::Dict, 
                             filename::String)

    nx, ny, nz = dims
    n_cells = length(density)
    n_points = (nx + 1) * (ny + 1) * (nz + 1)
    
    if !endswith(filename, ".vti"); filename *= ".vti"; end

    out_settings = get(config, "output_settings", Dict())
    
    save_stress_vec_val = get(out_settings, "save_principal_stress_vectors", "no")
    write_stress_vectors = (lowercase(string(save_stress_vec_val)) == "yes" || save_stress_vec_val == true)

    save_disp_vec_val = get(out_settings, "save_displacement_vectors", "yes")
    write_disp_vectors = (lowercase(string(save_disp_vec_val)) == "yes" || save_disp_vec_val == true)

    has_displacement = write_disp_vectors && (length(nodal_displacement) == n_points * 3)

    open(filename, "w") do io
        write(io, "<?xml version=\"1.0\"?>\n")
        write(io, "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\">\n")
        
        extent = "0 $nx 0 $ny 0 $nz"
        dx, dy, dz = spacing
        ox, oy, oz = origin
        
        write(io, "  <ImageData WholeExtent=\"$extent\" Origin=\"$ox $oy $oz\" Spacing=\"$dx $dy $dz\">\n")
        write(io, "    <Piece Extent=\"$extent\">\n")
        
        current_offset = 0
        
        # --- CELL DATA ---
        write(io, "      <CellData Scalars=\"Density\">\n")
        
        write(io, "        <DataArray type=\"Float32\" Name=\"Density\" format=\"appended\" offset=\"$current_offset\"/>\n")
        current_offset += sizeof(UInt32) + n_cells * sizeof(Float32)
        
        write(io, "        <DataArray type=\"Float32\" Name=\"L1_Stress\" format=\"appended\" offset=\"$current_offset\"/>\n")
        current_offset += sizeof(UInt32) + n_cells * sizeof(Float32)
        
        write(io, "        <DataArray type=\"Float32\" Name=\"VonMises\" format=\"appended\" offset=\"$current_offset\"/>\n")
        current_offset += sizeof(UInt32) + n_cells * sizeof(Float32)

        if write_stress_vectors
            write(io, "        <DataArray type=\"Float32\" Name=\"PrincipalValues\" NumberOfComponents=\"3\" format=\"appended\" offset=\"$current_offset\"/>\n")
            current_offset += sizeof(UInt32) + (n_cells * 3) * sizeof(Float32)

            write(io, "        <DataArray type=\"Float32\" Name=\"MaxPrincipalDirection\" NumberOfComponents=\"3\" format=\"appended\" offset=\"$current_offset\"/>\n")
            current_offset += sizeof(UInt32) + (n_cells * 3) * sizeof(Float32)

            write(io, "        <DataArray type=\"Float32\" Name=\"MinPrincipalDirection\" NumberOfComponents=\"3\" format=\"appended\" offset=\"$current_offset\"/>\n")
            current_offset += sizeof(UInt32) + (n_cells * 3) * sizeof(Float32)
        end
        write(io, "      </CellData>\n")

        # --- POINT DATA ---
        if has_displacement
            write(io, "      <PointData Vectors=\"Displacement\">\n")
            write(io, "        <DataArray type=\"Float32\" Name=\"Displacement\" NumberOfComponents=\"3\" format=\"appended\" offset=\"$current_offset\"/>\n")
            current_offset += sizeof(UInt32) + (n_points * 3) * sizeof(Float32)
            write(io, "      </PointData>\n")
        end

        write(io, "    </Piece>\n")
        write(io, "  </ImageData>\n")
        
        # --- APPENDED DATA ---
        # CRITICAL FIX: No newline between tag and marker.
        # This prevents an extra 0x0A byte from shifting the binary stream.
        write(io, "  <AppendedData encoding=\"raw\">") 
        write(io, "_") 
        
        function write_array(arr)
            n_bytes = UInt32(length(arr) * sizeof(Float32))
            write(io, n_bytes)
            write(io, arr)
        end

        write_array(density)
        write_array(l1_stress)
        write_array(von_mises)

        if write_stress_vectors
            write_array(vec(principal_vals)) 
            write_array(vec(principal_max_dirs))
            write_array(vec(principal_min_dirs))
        end

        if has_displacement
            write_array(vec(nodal_displacement))
        end
        
        write(io, "\n  </AppendedData>\n")
        write(io, "</VTKFile>\n")
    end
end

function export_solution_legacy(nodes::Matrix{Float32}, 
                                elements::Matrix{Int}, 
                                U_full::Vector{Float32}, 
                                F::Vector{Float32}, 
                                bc_indicator::Matrix{Float32}, 
                                principal_field::Matrix{Float32}, 
                                vonmises_field::Vector{Float32}, 
                                full_stress_voigt::Matrix{Float32}, 
                                l1_stress_norm_field::Vector{Float32},
                                principal_max_dir_field::Matrix{Float32}; 
                                density::Vector{Float32}=Float32[],
                                filename::String="solution.vtk",
                                kwargs...)
    println("Legacy exporter triggered.")
end

function export_solution(nodes, elements, U, F, bc, p_field, vm, voigt, l1, p_max_dir, p_min_dir; 
                         density=nothing, filename="out.vtk", config=nothing, kwargs...)
                           
    if config !== nothing
        geom = config["geometry"]
        nx = Int(geom["nElem_x_computed"])
        ny = Int(geom["nElem_y_computed"])
        nz = Int(geom["nElem_z_computed"])
        dx = Float32(geom["dx_computed"])
        dy = Float32(geom["dy_computed"])
        dz = Float32(geom["dz_computed"])
        
        out_settings = get(config, "output_settings", Dict())
        save_disp_vec_val = get(out_settings, "save_displacement_vectors", "yes")
        write_disp_vectors = (lowercase(string(save_disp_vec_val)) == "yes" || save_disp_vec_val == true)

        disp_matrix = zeros(Float32, 0, 0)

        if write_disp_vectors
            n_nodes_total = div(length(U), 3)
            disp_matrix = reshape(U, 3, n_nodes_total)
        end

        export_solution_vti((nx, ny, nz), (dx, dy, dz), (0f0, 0f0, 0f0), 
                            density, l1, vm, disp_matrix, p_field, p_max_dir, p_min_dir, config, filename)
    else
        export_solution_legacy(nodes, elements, U, F, bc, p_field, vm, voigt, l1, p_max_dir; 
                               density=density, filename=filename, kwargs...)
    end
end

end