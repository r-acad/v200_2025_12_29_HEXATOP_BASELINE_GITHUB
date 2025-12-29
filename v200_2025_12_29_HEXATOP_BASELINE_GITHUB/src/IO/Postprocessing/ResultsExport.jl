# FILE: .\src\IO\Postprocessing\ResultsExport.jl

using Printf
using ..ExportVTK
using ..Diagnostics

"""
    update_pvd_file(pvd_path, vti_filename, iter)

Updates a .pvd file. Uses purely relative paths for the VTI reference
to ensure the folder can be moved without breaking the animation.
"""
function update_pvd_file(pvd_path::String, vti_full_path::String, iter::Int)
    is_new = !isfile(pvd_path)
    lines = String[]
    
    if !is_new
        try
            lines = readlines(pvd_path)
            filter!(l -> !occursin("</Collection>", l) && !occursin("</VTKFile>", l), lines)
        catch
            is_new = true
        end
    end

    if is_new
        lines = [
            "<?xml version=\"1.0\"?>",
            "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">",
            "  <Collection>"
        ]
    end

    # Use only the filename, assuming .pvd and .vti are in the same folder.
    # This prevents absolute path issues.
    base_vti = basename(vti_full_path)
    
    entry = "    <DataSet timestep=\"$iter\" group=\"\" part=\"0\" file=\"$base_vti\"/>"
    push!(lines, entry)
    push!(lines, "  </Collection>")
    push!(lines, "</VTKFile>")

    open(pvd_path, "w") do io
        for line in lines
            println(io, line)
        end
    end
end

function export_iteration_results(iter::Int, base_name::String, RESULTS_DIR::String, 
                                  nodes::Matrix{Float32}, elements::Matrix{Int}, 
                                  U_full::AbstractVector, F::AbstractVector, 
                                  bc_indicator::Matrix{Float32}, principal_field::Matrix{Float32}, 
                                  vonmises_field::Vector{Float32}, full_stress_voigt::Matrix{Float32}, 
                                  l1_stress_norm_field::Vector{Float32}, principal_max_dir_field::Matrix{Float32}, principal_min_dir_field::Matrix{Float32}, 
                                  density::Vector{Float32}, E::Float32, geom; 
                                  iso_threshold::Float32=0.8f0, 
                                  current_radius::Float32=0.0f0, 
                                  integral_error::Float32=0.0f0, 
                                  config::Dict=Dict(), 
                                  save_bin::Bool=true, 
                                  save_stl::Bool=true, 
                                  save_vtk::Bool=true)
      
    U_f32 = (eltype(U_full) == Float32) ? U_full : Float32.(U_full)
    F_f32 = (eltype(F) == Float32) ? F : Float32.(F)

    iter_pad = lpad(iter, 4, '0')
    
    out_settings = get(config, "output_settings", Dict())
    raw_max_cells = get(out_settings, "maximum_cells_in_binary_output", 25_000_000)
    max_export_cells = safe_parse_int(raw_max_cells, 25_000_000)

    raw_stl_target = get(out_settings, "stl_target_triangle_count", 0)
    target_triangles = safe_parse_int(raw_stl_target, 0)

    if save_bin
        try
            print("      > Writing Checkpoint/Web Binary...")
            t_web = time()
            bin_filename = joinpath(RESULTS_DIR, "$(base_name)_webdata.$(iter_pad).bintop")
            export_binary_for_web(
                bin_filename, nodes, elements, density, l1_stress_norm_field, 
                principal_field, geom, iso_threshold, iter, current_radius, 
                integral_error, config; max_export_cells=max_export_cells
            )
            @printf(" done (%.3fs)\n", time() - t_web)
        catch e
            Diagnostics.print_warn("Web binary export failed. Continuing.")
            Diagnostics.write_crash_log("crash_log.txt", "WEB_EXPORT", e, stacktrace(catch_backtrace()), iter, config, density)
        end
    end

    if save_vtk
        try
            print("      > Writing VTK (Paraview)...")
            t_vtk = time()
            
            solution_filename = joinpath(RESULTS_DIR, "$(base_name)_solution.$(iter_pad).vti") 
            
            ExportVTK.export_solution(nodes, elements, U_f32, F_f32, bc_indicator, 
                                      principal_field, vonmises_field, full_stress_voigt, 
                                      l1_stress_norm_field, principal_max_dir_field, principal_min_dir_field; 
                                      density=density, threshold=iso_threshold, scale=Float32(1.0), 
                                      filename=solution_filename,
                                      config=config, 
                                      max_cells=0)

            # Update PVD for animation
            pvd_filename = joinpath(RESULTS_DIR, "$(base_name)_animation.pvd")
            update_pvd_file(pvd_filename, solution_filename, iter)

            @printf(" done (%.3fs)\n", time() - t_vtk)
        catch e
            Diagnostics.print_warn("VTK export failed. Continuing.")
            Diagnostics.write_crash_log("crash_log.txt", "VTK_EXPORT", e, stacktrace(catch_backtrace()), iter, config, density)
        end
    end

    if save_stl && iter > 0 
        print("      > Writing Isosurface STL...")
        t_stl = time()
        stl_filename = joinpath(RESULTS_DIR, "$(base_name)_isosurface.$(iter_pad).stl")
        
        subdiv = get(out_settings, "stl_subdivision_level", 2)
        smooth = get(out_settings, "stl_smoothing_passes", 2)
        mesh_smooth = get(out_settings, "stl_mesh_smoothing_iters", 3)
        
        export_smooth_watertight_stl(density, geom, iso_threshold, stl_filename; 
                                     subdivision_level=subdiv, 
                                     smoothing_passes=smooth,
                                     mesh_smoothing_iters=mesh_smooth,
                                     target_triangle_count=target_triangles)
        @printf(" done (%.3fs)\n", time() - t_stl)
    end
end