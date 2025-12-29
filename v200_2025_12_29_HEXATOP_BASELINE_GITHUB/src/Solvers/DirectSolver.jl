# FILE: .\src\Solvers\DirectSolver.jl
module DirectSolver

using LinearAlgebra
using SparseArrays
using Base.Threads
using Printf
using ..Element
using ..Diagnostics # Using the common diagnostics module for logging

export solve_system

"""
    assemble_global_stiffness_parallel(nodes, elements, E, nu, density, min_stiffness)

Optimized parallel assembly of the global stiffness matrix.
Filters inactive elements to save memory and time.
"""
function assemble_global_stiffness_parallel(nodes::Matrix{Float32},
                                            elements::Matrix{Int},
                                            E::Float32,
                                            nu::Float32,
                                            density::Vector{Float32},
                                            min_stiffness_threshold::Float32)
                                            
    nElem = size(elements, 1)
    nNodes = size(nodes, 1)
    ndof = nNodes * 3

    # Filter for active elements to reduce work
    active_indices = findall(d -> d >= min_stiffness_threshold, density)
    nActive = length(active_indices)
    
    if nActive == 0; error("DirectSolver: No active elements found."); end

    # Compute canonical stiffness matrix once (assuming uniform grid)
    # n1, n2, n4, n5 correspond to local indices 1, 2, 4, 5
    n1_idx = elements[1, 1]; n2_idx = elements[1, 2]
    n4_idx = elements[1, 4]; n5_idx = elements[1, 5]
    
    dx = norm(nodes[n2_idx, :] - nodes[n1_idx, :])
    dy = norm(nodes[n4_idx, :] - nodes[n1_idx, :])
    dz = norm(nodes[n5_idx, :] - nodes[n1_idx, :])
    
    Ke_base = Element.get_canonical_stiffness(dx, dy, dz, nu)

    entries_per_elem = 576 # 24*24
    total_entries = nActive * entries_per_elem
    
    # Pre-allocate triplet arrays
    I_vec = Vector{Int32}(undef, total_entries)
    J_vec = Vector{Int32}(undef, total_entries)
    V_vec = Vector{Float32}(undef, total_entries)

    Diagnostics.print_substep("Direct Solver: Assembling $nActive active elements (Parallel)...")

    # Parallel Assembly Loop [Optimized from your version]
    Threads.@threads for t_idx in 1:nActive
        e = active_indices[t_idx]
        offset = (t_idx - 1) * entries_per_elem
        
        # Linear stiffness scaling: E_loc = E * density
        factor = E * density[e]
        
        conn = view(elements, e, :)
        
        cnt = 0
        @inbounds for i in 1:8
            row_node = conn[i]
            row_base = 3 * (row_node - 1)
            
            for r in 1:3
                g_row = row_base + r
                
                for j in 1:8
                    col_node = conn[j]
                    col_base = 3 * (col_node - 1)
                    
                    for c in 1:3
                        g_col = col_base + c
                        cnt += 1
                        
                        idx = offset + cnt
                        I_vec[idx] = Int32(g_row)
                        J_vec[idx] = Int32(g_col)
                        V_vec[idx] = Ke_base[3*(i-1)+r, 3*(j-1)+c] * factor
                    end
                end
            end
        end
    end

    # Create sparse matrix
    # Note: sparse() automatically sums duplicate entries if any exist
    K_global = sparse(I_vec, J_vec, V_vec, ndof, ndof)
    
    # Enforce symmetry to fix potential floating point drift and ensure LDLt/Cholesky usage
    return (K_global + K_global') / 2.0f0
end

"""
    solve_system(...)

Main entry point for the Direct Solver. 
Integrates assembly, BC application, and linear solve.
"""
function solve_system(nodes::Matrix{Float32}, 
                      elements::Matrix{Int}, 
                      E::Float32, 
                      nu::Float32,
                      bc_indicator::Matrix{Float32}, 
                      F::Vector{Float32};
                      density::Vector{Float32}=Float32[],
                      config::Dict=Dict(), # Matches Solver.jl interface
                      shift_factor::Float32=1.0e-7,
                      min_stiffness_threshold::Float32=1.0e-6,
                      kwargs...)
    
    #  can be visualized here conceptually if needed
                  
    nElem = size(elements, 1)
    if isempty(density); density = ones(Float32, nElem); end

    nNodes = size(nodes, 1)
    ndof = nNodes * 3

    # 1. Identify Free DoFs (Reduction Method)
    constrained = falses(ndof)
    for i in 1:nNodes
        if bc_indicator[i,1] > 0; constrained[3*(i-1)+1] = true; end
        if bc_indicator[i,2] > 0; constrained[3*(i-1)+2] = true; end
        if bc_indicator[i,3] > 0; constrained[3*(i-1)+3] = true; end
    end
    free_dofs = findall(!, constrained)

    # 2. Assemble Global K
    K_global = assemble_global_stiffness_parallel(nodes, elements, E, nu, density, min_stiffness_threshold)
    
    # CRITICAL: Clean up massive triplet arrays immediately
    GC.gc() 

    # 3. System Reduction
    Diagnostics.print_substep("Direct Solver: Reducing System...")
    K_reduced = K_global[free_dofs, free_dofs]
    F_reduced = F[free_dofs]
    
    # Free global K memory as it's no longer needed
    K_global = nothing
    GC.gc()

    # 4. Diagonal Shift (Regularization)
    # This prevents the solver from crashing if 'floating' material exists
    try
        diag_vals = diag(K_reduced)
        avg_diag = mean(abs.(diag_vals))
        shift = shift_factor * avg_diag
        
        Diagnostics.print_substep("Direct Solver: Applying diagonal shift: $(@sprintf("%.2e", shift))")
        
        # Efficient diagonal addition
        K_reduced = K_reduced + shift * I
    catch e
        Diagnostics.print_warn("Could not apply diagonal shift: $e")
    end
    
    # 5. Solve
    Diagnostics.print_substep("Direct Solver: Factorizing (LDLt/LU)...")
    t_solve = time()
    
    # The '\' operator is polyalgorithmic. Since we symmetrized K, 
    # it will typically use LDLt or Cholesky, which are optimal.
    U_reduced = K_reduced \ F_reduced
    
    solve_time = time() - t_solve
    Diagnostics.print_substep("Direct Solver: Done in $(round(solve_time, digits=3))s")

    # 6. Reconstruct Full Vector
    U_full = zeros(Float32, ndof)
    U_full[free_dofs] = U_reduced

    return (U_full, 0.0, "Direct_CPU")
end

end