# FILE: .\src\Optimization\Verification.jl

using Printf

function verify_boundary_filtering_detailed(density::Vector{Float32}, filtered::Vector{Float32}, 
                                            nx::Int, ny::Int, nz::Int)
    
    interior_changed = 0; interior_total = 0
    faces_changed = 0; faces_total = 0
    edges_changed = 0; edges_total = 0
    corners_changed = 0; corners_total = 0
    
    nElem = length(density)

    for k in 1:nz, j in 1:ny, i in 1:nx
        e = i + (j-1)*nx + (k-1)*nx*ny
        
        changed = abs(density[e] - filtered[e]) > 1e-6
        
        on_boundary_count = 0
        if i == 1 || i == nx; on_boundary_count += 1; end
        if j == 1 || j == ny; on_boundary_count += 1; end
        if k == 1 || k == nz; on_boundary_count += 1; end
        
        if on_boundary_count == 0
            interior_total += 1
            if changed; interior_changed += 1; end
        elseif on_boundary_count == 1
            faces_total += 1
            if changed; faces_changed += 1; end
        elseif on_boundary_count == 2
            edges_total += 1
            if changed; edges_changed += 1; end
        else  
            corners_total += 1
            if changed; corners_changed += 1; end
        end
    end
    
    println("    [Filter Check] Detailed Boundary Analysis:")
    
    if interior_total > 0
        pct = 100.0 * interior_changed / interior_total
        status = pct > 90.0 ? "✓" : "✗"
        println(@sprintf("      %s Interior:  %6d / %6d (%.1f%%)", 
                         status, interior_changed, interior_total, pct))
    end
    
    if faces_total > 0
        pct = 100.0 * faces_changed / faces_total
        status = pct > 90.0 ? "✓" : "✗"
        println(@sprintf("      %s Faces:       %6d / %6d (%.1f%%)", 
                         status, faces_changed, faces_total, pct))
    end
    
    if edges_total > 0
        pct = 100.0 * edges_changed / edges_total
        status = pct > 80.0 ? "✓" : "✗"
        println(@sprintf("      %s Edges:       %6d / %6d (%.1f%%)", 
                         status, edges_changed, edges_total, pct))
    end
    
    if corners_total > 0
        pct = 100.0 * corners_changed / corners_total
        status = pct > 70.0 ? "✓" : "✗"
        println(@sprintf("      %s Corners:     %6d / %6d (%.1f%%)", 
                         status, corners_changed, corners_total, pct))
    end
    
    total_boundary = faces_total + edges_total + corners_total
    total_boundary_changed = faces_changed + edges_changed + corners_changed
    
    if total_boundary > 0
        overall_pct = 100.0 * total_boundary_changed / total_boundary
        if overall_pct < 80.0
            println("    \u001b[33m[WARNING] <80%% of boundaries filtered properly!\u001b[0m")
        else
            println("    \u001b[32m[SUCCESS] Boundary filtering working correctly.\u001b[0m")
        end
    end
end