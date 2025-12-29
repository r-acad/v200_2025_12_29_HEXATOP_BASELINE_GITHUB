// # FILE: .\src\App\SimulationLoop.jl";


module SimulationLoop

using LinearAlgebra
using Printf
using Base.Threads
using Dates
using Statistics
using CUDA


using ..Diagnostics
using ..Helpers
using ..Configuration
using ..Boundary
using ..Element
using ..Stress
using ..Mesh
using ..MeshRefiner
using ..Solver
using ..Postprocessing
using ..TopologyOptimization
using ..HardwareProfile

export run_simulation

function run_simulation(current_config::Dict, run_name::String="Simulation"; project_root::String="")
    Diagnostics.print_banner("HEXA TOPOLOGY OPTIMIZER: $run_name")
    Diagnostics.print_info("Clearing GPU Memory from previous runs...")
    
    if CUDA.functional()
        Helpers.clear_gpu_memory()
        CUDA.device!(0) 
        dev = CUDA.device()
        name = CUDA.name(dev)
        total_mem = CUDA.total_memory()
        mem_gb = total_mem / 1024^3
        Diagnostics.print_success("GPU Detected: $name ($(round(mem_gb, digits=2)) GB)")
    end
    GC.gc()
    
    HardwareProfile.apply_hardware_profile!(current_config)
    
    restart_conf = get(current_config, "restart_configuration", Dict())
    enable_restart = get(restart_conf, "enable_restart", false)
    restart_path = get(restart_conf, "file_path", "")
    
    config = Dict{Any,Any}()
    density = Float32[]
    start_iter = 1
    restart_radius = 0.0f0
    restart_threshold = 0.0f0
    integral_error = 0.0f0
    U_full = Float32[] 
    is_restart_active = false

    if enable_restart && isfile(restart_path)
         Diagnostics.print_banner("RESTART MODE ACTIVE", color="\u001b[35m")
         saved_config, density_dummy, U_loaded, i_err, saved_iter, restart_radius, restart_threshold = Configuration.load_checkpoint(restart_path)
         config = merge(saved_config, current_config)
         HardwareProfile.apply_hardware_profile!(config)
         start_iter = saved_iter + 1
         integral_error = i_err
         is_restart_active = true
    else
         config = current_config
         is_restart_active = false
    end
    
    hard_stop_iter = get(config, "hard_stop_after_iteration", -1)

    out_settings = get(config, "output_settings", Dict())
    default_freq = get(out_settings, "export_frequency", 5)
    save_bin_freq = get(out_settings, "save_bin_frequency", default_freq)
    save_stl_freq = get(out_settings, "save_STL_frequency", default_freq)
    save_vtk_freq = get(out_settings, "save_VTK_frequency", default_freq)

    save_vec_val = get(out_settings, "save_principal_stress_vectors", "no")
    save_vectors_bool = (lowercase(string(save_vec_val)) == "yes" || save_vec_val == true)
    
    RESULTS_DIR = joinpath(project_root, "RESULTS", run_name)
    if !isdir(RESULTS_DIR); mkpath(RESULTS_DIR); end
    Diagnostics.print_info("Output Directory: $RESULTS_DIR")

    raw_log_name = get(out_settings, "log_filename", "simulation_log.txt")
    log_base, log_ext = splitext(basename(raw_log_name))
    log_filename = joinpath(RESULTS_DIR, "$(log_base)_$(run_name)$(log_ext)")
    
    iso_threshold_val = get(out_settings, "iso_surface_threshold", 0.01)
    iso_threshold = Float32(iso_threshold_val)
    
    if !is_restart_active
        Diagnostics.init_log_file(log_filename, config)
    else
        Diagnostics.log_status("--- RESTARTING SIMULATION (Iter $start_iter) ---")
    end
    
    geom = Configuration.setup_geometry(config)
    nodes, elements, dims = generate_mesh(geom.nElem_x, geom.nElem_y, geom.nElem_z; dx = geom.dx, dy = geom.dy, dz = geom.dz)
    initial_target_count = size(elements, 1)
    
    domain_bounds = (min_pt=[0.0f0,0.0f0,0.0f0], len_x=geom.dx*geom.nElem_x, len_y=geom.dy*geom.nElem_y, len_z=geom.dz*geom.nElem_z)
    config["geometry"]["nElem_x_computed"] = geom.nElem_x
    config["geometry"]["nElem_y_computed"] = geom.nElem_y
    config["geometry"]["nElem_z_computed"] = geom.nElem_z
    config["geometry"]["dx_computed"] = geom.dx
    config["geometry"]["dy_computed"] = geom.dy
    config["geometry"]["dz_computed"] = geom.dz
    config["geometry"]["max_domain_dim"] = geom.max_domain_dim
    
    nNodes = size(nodes, 1)
    ndof = nNodes * 3
    
    if length(U_full) != ndof
        U_full = zeros(Float32, ndof)
    end

    bc_data = config["boundary_conditions"]
    forces_data = config["external_forces"]
    
    bc_indicator = Boundary.get_bc_indicator(nNodes, nodes, Vector{Any}(bc_data))
    F_external = zeros(Float32, ndof)
    Boundary.apply_external_forces!(F_external, Vector{Any}(forces_data), nodes, elements)
    Diagnostics.print_success("Boundary Conditions & External Forces Mapped.")

    E = Float32(config["material"]["E"])
    nu = Float32(config["material"]["nu"])
    
    
    # material_density = Float32(get(config["material"], "material_density", 0.0)) 
    
    gravity_accel = Float32(get(config["material"], "gravity_acceleration", 9.81))
    delta_T = Float32(get(config["material"], "delta_temperature", 0.0))
    if abs(delta_T) > 1e-6; Diagnostics.print_info("THERMOELASTICITY ENABLED: Delta T = $delta_T"); end

    original_density = ones(Float32, size(elements, 1)) 
    protected_elements_mask = falses(size(elements, 1)) 
    alpha_field = zeros(Float32, size(elements, 1))
    mass_density_field = zeros(Float32, size(elements, 1))

    
    density, original_density, protected_elements_mask, alpha_field, mass_density_field = Configuration.initialize_density_field(nodes, elements, geom.shapes, config)
    
    opt_params = config["optimization_parameters"]
    min_density = Float32(get(opt_params, "min_density", 1.0e-3))
    max_density_clamp = Float32(get(opt_params, "density_clamp_max", 1.0))
    max_culling_ratio = Float32(get(opt_params, "max_culling_ratio", 0.05))
    base_name = run_name 
    
    mesh_conf = get(config, "mesh_settings", Dict())
    
    nElem = size(elements, 1)
    estimated_iters = Helpers.estimate_required_iterations(config, nElem)
    
    nominal_iters_conf = get(config, "number_of_iterations", "AUTO")
    max_planned_iterations = 200 

    if nominal_iters_conf == "AUTO" || nominal_iters_conf == "auto"
        max_planned_iterations = estimated_iters
        Diagnostics.print_banner("AUTO-SCHEDULING")
        println("      > Calculated:        $estimated_iters iterations")
        println("      > Final Setting:  $max_planned_iterations iterations")
    else
        user_iters = Int(nominal_iters_conf)
        max_planned_iterations = user_iters
    end
    
    raw_active_target = get(mesh_conf, "final_target_of_active_elements", initial_target_count)
    final_target_active = isa(raw_active_target, String) ? parse(Int, replace(raw_active_target, "_" => "")) : Int(raw_active_target)
    max_growth_rate = Float64(get(mesh_conf, "max_growth_rate", 1.2))
    raw_bg_limit = get(mesh_conf, "max_background_elements", 800_000_000)
    hard_elem_limit = isa(raw_bg_limit, String) ? parse(Int, replace(raw_bg_limit, "_" => "")) : Int(raw_bg_limit)
    Diagnostics.print_info("Hard Element Limit: $(Base.format_bytes(hard_elem_limit * 100)) approx ($hard_elem_limit elems)")

    
    mat_config = config["material"]
    l1_stress_default = Float32(get(mat_config, "l1_stress_allowable", 1.0))
    
    limit_tension = Float32(get(mat_config, "l1_stress_allowable_tension", l1_stress_default))
    limit_compression = Float32(get(mat_config, "l1_stress_allowable_compression", l1_stress_default))
    
    Diagnostics.print_info("STRESS LIMITS:")
    println("      > Tension:      $limit_tension")
    println("      > Compression: $limit_compression")
    
    target_metric_val = 1.0f0 
    
    STABILITY_TARGET = Float64(get(opt_params, "convergence_stability_tolerance", 0.002))
    STRESS_ACCURACY_TARGET = Float64(get(opt_params, "convergence_stress_error_tolerance", 0.01))
    CONSECUTIVE_ITERS_REQUIRED = Int(get(opt_params, "convergence_required_streak", 5))
    
    update_damping_exp = Float32(get(opt_params, "update_damping_exponent", 1.0))
    
    Diagnostics.print_info("CONTROL MODE: Accelerated Stress Controller (Adaptive Cutoff)")
    
    
    internal_safety_factor = 1.0f0
    
    ctrl_conf = get(config, "stress_controller", Dict())
    Kp = Float32(get(ctrl_conf, "Kp", 0.02))
    Ki = Float32(get(ctrl_conf, "Ki", 0.005))
    integral_error_limit = Float32(get(ctrl_conf, "integral_error_limit", 50.0))
    max_adj_pct = Float32(get(ctrl_conf, "max_adjustment_pct", 0.05))
    state_switch_low = Float32(get(ctrl_conf, "state_switch_error_low", 0.05))
    state_switch_high = Float32(get(ctrl_conf, "state_switch_error_high", 0.15))
    min_var_switch = Float32(get(ctrl_conf, "min_variance_for_switch", 0.25))
    metric_percentile = Float64(get(ctrl_conf, "metric_percentile", 0.95))
    
    avg_element_size = (geom.dx + geom.dy + geom.dz) / 3.0f0
    target_d_phys = Float32(get(opt_params, "minimum_feature_size_physical", 0.0))
    floor_d_elems = Float32(get(opt_params, "minimum_feature_size_elements", 3.0)) 
    d_min_phys = max(target_d_phys, floor_d_elems * avg_element_size)
    
    current_radius = is_restart_active ? restart_radius : (d_min_phys * 4.0f0)
    
    starting_cutoff = Float32(get(opt_params, "starting_density_threshold", 0.10))
    current_cutoff = starting_cutoff
    
    R_min_val = d_min_phys * 1.0f0

    cutoff_state = 0 
    frozen_cutoff_counter = 0 
    current_stress_variance = 1.0f0
    
    iter = start_iter
    keep_running = true
    
    prev_density = copy(density)
    last_valid_density = copy(density) 
    prev_compliance = Inf 
    
    stability_metric = 1.0
    consecutive_converged_iters = 0
    
    relaxing_error_trend = 0.0f0
    prev_metric_error = 0.0f0

    Diagnostics.print_banner("STARTING MAIN LOOP")
    Diagnostics.print_info("Log File: $log_filename")
    
    flush(stdout) 

    while keep_running
        iter_start_time = time()
        status_msg = "Optimizing"
        
        cur_dims_str = "$(config["geometry"]["nElem_x_computed"])x$(config["geometry"]["nElem_y_computed"])x$(config["geometry"]["nElem_z_computed"])"
        config["current_outer_iter"] = iter
        
        current_active = count(d -> d > 0.01, density)
        nominal_ref_thresh = Float64(get(mesh_conf, "nominal_refinement_threshold", 0.8))
        
        if current_active < (final_target_active * nominal_ref_thresh) && size(elements, 1) < final_target_active
             prev_elem_count = size(elements, 1)
             nodes, elements, density, alpha_field, mass_density_field, dims = MeshRefiner.refine_mesh_and_fields(
                nodes, elements, density, alpha_field, mass_density_field, dims, final_target_active, domain_bounds;
                max_growth_rate = max_growth_rate, hard_element_limit = hard_elem_limit
            )
            GC.gc()
            
            if size(elements, 1) > prev_elem_count
                status_msg = "Refined"
                consecutive_converged_iters = 0
                current_dx = domain_bounds.len_x / (dims[1]-1)
                avg_element_size = current_dx 
                d_min_phys = max(target_d_phys, floor_d_elems * avg_element_size)
                config["geometry"]["nElem_x_computed"] = dims[1]-1
                config["geometry"]["nElem_y_computed"] = dims[2]-1
                config["geometry"]["nElem_z_computed"] = dims[3]-1
                config["geometry"]["dx_computed"] = current_dx
                config["geometry"]["dy_computed"] = domain_bounds.len_y / (dims[2]-1)
                config["geometry"]["dz_computed"] = domain_bounds.len_z / (dims[3]-1)
                
                geom = (nElem_x=dims[1]-1, nElem_y=dims[2]-1, nElem_z=dims[3]-1, 
                        dx=config["geometry"]["dx_computed"], dy=config["geometry"]["dy_computed"], dz=config["geometry"]["dz_computed"], 
                        shapes=geom.shapes, actual_elem_count=size(elements, 1), max_domain_dim=geom.max_domain_dim)

                Diagnostics.print_substep("[Refinement] Re-mapping Boundary Conditions & Forces...")
                nNodes = size(nodes, 1)
                ndof = nNodes * 3
                bc_indicator = Boundary.get_bc_indicator(nNodes, nodes, Vector{Any}(config["boundary_conditions"]))
                F_external = zeros(Float32, ndof)
                Boundary.apply_external_forces!(F_external, Vector{Any}(config["external_forces"]), nodes, elements)
                
                
                _, original_density, protected_elements_mask, _, _ = Configuration.initialize_density_field(nodes, elements, geom.shapes, config)
                
                U_full = zeros(Float32, ndof)
                TopologyOptimization.reset_filter_cache!()
                prev_density = copy(density)
                last_valid_density = copy(density) 
                prev_compliance = Inf
                cutoff_state = 0 
                relaxing_error_trend = 0.0f0 
            end
        end

        Threads.@threads for e in 1:size(elements, 1)
            if protected_elements_mask[e]; density[e] = original_density[e]; end
        end
        
        F_total = copy(F_external)
        
        
        if gravity_accel != 0.0
             Boundary.add_self_weight!(F_total, density, mass_density_field, protected_elements_mask, 1.0f0, elements, geom.dx, geom.dy, geom.dz, gravity_accel)
        end
        
        
        if abs(delta_T) > 1e-6
             Boundary.compute_global_thermal_forces!(F_total, nodes, elements, alpha_field, delta_T, E, nu, density)
        end
        
        C_ORANGE = "\u001b[38;5;208m"
        C_RESET = "\u001b[0m"
        C_YELLOW = "\u001b[33m"
        Diagnostics.print_substep("FEA Solve (Iter " * C_ORANGE * "$iter / $max_planned_iterations" * C_RESET * ")")
        
        sol_tuple = Solver.solve_system(
            nodes, elements, E, nu, bc_indicator, F_total;
            density=density, config=config, min_stiffness_threshold=min_density, 
            prune_voids=true, u_prev=U_full 
        )
        U_new = sol_tuple[1]
        last_residual = sol_tuple[2]
        prec_used = sol_tuple[3]
        U_full = U_new
        
        if CUDA.functional(); GC.gc(); CUDA.reclaim(); end
        
        compliance = dot(F_total, U_full)
        strain_energy = 0.5 * compliance
        
        
        elem_vol = geom.dx * geom.dy * geom.dz
        
        
        
        current_weight = 0.0
        for e in 1:length(density)
            if protected_elements_mask[e]
                current_weight += mass_density_field[e] * elem_vol
            else
                current_weight += density[e] * mass_density_field[e] * elem_vol
            end
        end
        
        # === REVISION: Lowered safety guard from 5 to 1 ===
        # Previously, bad solves (noise) in iterations 1-5 were ignored, causing the controller to 
        # wind up and only stabilize at iter 7 or 8. Now we catch them immediately.
        if iter > 1 && compliance > (prev_compliance * 1.5)
            println(C_YELLOW * ">>> [BACKTRACK] Compliance spike detected. Stabilizing..." * C_RESET)
            density .= last_valid_density
            internal_safety_factor *= 0.75f0 
            integral_error = 0.0f0 
            cutoff_state = 0 
            relaxing_error_trend = 0.0f0 
            
            Diagnostics.write_iteration_log(
                log_filename, iter, cur_dims_str, length(density), current_active, 
                current_radius, current_cutoff, compliance, strain_energy, 0.0, 
                0.0, 0.0, "Backtrack", 0.0, last_residual, prec_used,
                stability_metric, 0.0, 0, 0.0, current_weight
            )
            iter += 1
            continue 
        end
        
        last_valid_density .= density
        prev_compliance = compliance
        
        Diagnostics.print_substep("Calculating Stress Field...")
        
        
        principal_field, vonmises_field, full_stress_voigt, l1_stress_norm_field, principal_max_dir_field, principal_min_dir_field = Stress.compute_stress_field(nodes, elements, U_full, E, nu, density; return_voigt=false)
        
        try
            if iter == 1
                Diagnostics.print_info("Exporting INITIAL REFERENCE STATE (Iter 0)...")
                do_bin_init = (save_bin_freq > 0); do_stl_init = (save_stl_freq > 0); do_vtk_init = (save_vtk_freq > 0)
                if do_bin_init || do_stl_init || do_vtk_init
                    Postprocessing.export_iteration_results(0, base_name, RESULTS_DIR, nodes, elements, U_full, F_total, bc_indicator, 
                                                            principal_field, vonmises_field, full_stress_voigt, l1_stress_norm_field, 
                                                            principal_max_dir_field, principal_min_dir_field, 
                                                            density, E, geom; 
                                                            iso_threshold=Float32(iso_threshold), 
                                                            current_radius=Float32(current_radius), 
                                                            integral_error=Float32(integral_error),
                                                            config=config, 
                                                            save_bin=do_bin_init, save_stl=do_stl_init, save_vtk=do_vtk_init)
                end
                if hard_stop_iter == 0; println(">>> HARD STOP: Stopping after background analysis (Iter 0)."); keep_running = false; break; end
            end
        catch e_export
            Diagnostics.print_warn("Initial export failed ($e_export).")
        end
        
        active_indices = findall(i -> density[i] > 0.1f0 && !protected_elements_mask[i], 1:length(density))
        
        current_metric_val = 0.0f0
        current_variance = 0.0f0
        
        if !isempty(active_indices)
            
            utilization_values = Vector{Float32}(undef, length(active_indices))
            
            for (i, idx) in enumerate(active_indices)
                raw_s = l1_stress_norm_field[idx]
                
                
                lim = (raw_s >= 0) ? limit_tension : limit_compression
                utilization_values[i] = abs(raw_s) / abs(lim)
            end
            
            sort!(utilization_values)
            n_vals = length(utilization_values)
            cutoff_idx = floor(Int, n_vals * metric_percentile)
            
            if cutoff_idx > 10
                subset = view(utilization_values, 1:cutoff_idx)
                current_metric_val = mean(subset)
                std_val = std(subset)
                if current_metric_val > 1e-6
                    current_variance = std_val / current_metric_val
                end
            else
                current_metric_val = 0.0f0
            end
        end

        
        metric_error = (current_metric_val - target_metric_val) / target_metric_val
        
        if cutoff_state == 0 
            if abs(metric_error) < state_switch_low
                cutoff_state = 1
                relaxing_error_trend = 0.0f0
            elseif (abs(metric_error) < state_switch_high) && (current_variance > min_var_switch)
                cutoff_state = 1
                relaxing_error_trend = 0.0f0
            end
        elseif cutoff_state == 1 
            if abs(metric_error) > 1.0f0
                cutoff_state = 0 
                relaxing_error_trend = 0.0f0
            end
            
            if abs(metric_error) > prev_metric_error
                relaxing_error_trend += 1.0f0
            else
                relaxing_error_trend = max(0.0f0, relaxing_error_trend - 0.5f0)
            end
            
            if relaxing_error_trend >= 5.0f0
                Diagnostics.print_warn(">>> [CONTROLLER] Relaxing Failed (Error Rising). Backing off.")
                cutoff_state = 0 
                current_cutoff = max(starting_cutoff, current_cutoff * 0.95f0) 
                integral_error *= 0.5f0 
                relaxing_error_trend = 0.0f0
            end
        end
        prev_metric_error = abs(metric_error)
        
        if cutoff_state == 1
            frozen_cutoff_counter += 1
            status_msg = "Relaxing ($frozen_cutoff_counter)"
        else
            frozen_cutoff_counter = 0
        end
        
        control_error = -metric_error 
        integral_error += control_error
        
        integral_error = clamp(integral_error, -integral_error_limit, integral_error_limit) 
        
        eff_Kp = (cutoff_state == 1) ? (Kp * 0.1f0) : Kp
        eff_Ki = Ki * 0.1f0
        
        adjustment_pct = (eff_Kp * control_error) + (eff_Ki * integral_error)
        
        adjustment_pct = clamp(adjustment_pct, -max_adj_pct, max_adj_pct)
        
        adjustment_factor = 1.0f0 + adjustment_pct
        internal_safety_factor *= adjustment_factor
        internal_safety_factor = clamp(internal_safety_factor, 0.1f0, 10.0f0)

        final_cutoff_target = Float32(get(opt_params, "final_density_threshold", 0.5))
        progress = Float32(iter) / Float32(max_planned_iterations)
        progress = clamp(progress, 0.0f0, 1.0f0)
        
        accel_factor = 1.0f0
        if metric_error < -0.2f0  
            accel_factor = 2.0f0  
        end
        
        base_cutoff_sched = starting_cutoff 
        cutoff_exponent = Float32(get(opt_params, "exponent_for_cutoff_schedule", 0.7)) 
        effective_progress = clamp(progress * accel_factor, 0.0f0, 1.0f0)
        target_scheduled_cutoff = base_cutoff_sched + (final_cutoff_target - base_cutoff_sched) * (effective_progress ^ cutoff_exponent) 
        
        if cutoff_state == 0 
            rise_limit = (metric_error < -0.2f0) ? 0.01f0 : 0.002f0
            if abs(metric_error) < 0.10f0; rise_limit = 0.0002f0; end
            current_cutoff = min(target_scheduled_cutoff, current_cutoff + rise_limit)
        end
        
        R_sched_val = d_min_phys * 4.0f0 + (d_min_phys * 1.0f0 - d_min_phys * 4.0f0) * ((progress)^2)
        
        if cutoff_state == 1
            current_radius = R_min_val
        else
            current_radius = R_sched_val
        end

        Diagnostics.print_substep("Controller Status:")
        println("      > Utilization:          $(round(current_metric_val * 100, digits=2))% (Target: 100%)")
        println("      > Safety Factor:        $(round(internal_safety_factor, digits=3))")
        println("      > Error:                $(round(metric_error * 100, digits=2))%")
        println("      > Cutoff Status:        $(cutoff_state == 1 ? "FROZEN (Relaxing)" : "RISING")")
        println("      > Current Cutoff:       $(round(current_cutoff, digits=3))")

        Diagnostics.print_substep("Topology Update & Filtering...")
        
        
        eff_limit_T = limit_tension * internal_safety_factor
        eff_limit_C = limit_compression * internal_safety_factor
        
        mean_change, _, actual_cutoff, filter_time, _, _ = TopologyOptimization.update_density!(
            density, l1_stress_norm_field, protected_elements_mask, E, eff_limit_T, eff_limit_C, 
            iter, max_planned_iterations, 
            original_density, min_density, max_density_clamp, config, elements; 
            force_no_cull = false,
            cutoff_threshold = current_cutoff,        
            specified_radius = current_radius,        
            max_culling_ratio = max_culling_ratio,
            update_damping = update_damping_exp
        )
        
        active_indices = findall(x -> x > min_density, density)
        n_active = length(active_indices)
        
        if n_active > 0
            changed_elements = count(i -> abs(density[i] - prev_density[i]) > 0.01, active_indices)
            stability_metric = Float64(changed_elements) / Float64(n_active)
        else
            stability_metric = 0.0
        end
        
        stress_metric = abs(metric_error)
        radius_converged = (current_radius <= (R_min_val * 1.05))
        variance_metric = current_variance
        
        is_currently_converged = (stability_metric < 0.005) && 
                                 (stress_metric < 0.05) && 
                                 (variance_metric < 0.25) &&
                                 radius_converged

        if is_currently_converged
            consecutive_converged_iters += 1
        else
            consecutive_converged_iters = 0
        end
        
        println(C_YELLOW * "    [Topo Opt Iter: $iter/$max_planned_iterations]" * C_RESET)
        println("      > Stability:  $(round(stability_metric*100, digits=2))% (Target < 0.5%)")
        println("      > Stress Err: $(round(stress_metric*100, digits=2))% (Target < 5.0%)")
        println("      > Stress Var: $(round(variance_metric*100, digits=2))% (Target < 25.0%)")
        println("      > Radius End: $(radius_converged ? "YES" : "NO") ($(round(current_radius,digits=3)) vs $(round(R_min_val,digits=3)))")
        println("      > Streak:          $consecutive_converged_iters / $CONSECUTIVE_ITERS_REQUIRED steps")
        
        prev_density = copy(density)
        iter_time = time() - iter_start_time
        vol_total = length(density)
        vol_frac = sum(density)/vol_total
        
        Diagnostics.write_iteration_log(
            log_filename, iter, cur_dims_str, vol_total, n_active, 
            current_radius, current_cutoff, compliance, strain_energy, current_metric_val, 
            vol_frac, mean_change, status_msg, iter_time, last_residual, prec_used,
            stability_metric, stress_metric, consecutive_converged_iters, variance_metric,
            current_weight
        )

        is_last_iter = (!keep_running) || (hard_stop_iter > 0 && iter >= hard_stop_iter)
        
        if consecutive_converged_iters >= CONSECUTIVE_ITERS_REQUIRED
            Diagnostics.print_success("EARLY CONVERGENCE REACHED (Stable for $consecutive_converged_iters steps)!")
            keep_running = false
            is_last_iter = true
        end
        
        if keep_running && iter >= max_planned_iterations
            if max_planned_iterations < 1000 
                Diagnostics.print_warn("Target iterations reached but NOT converged. Extending simulation by 50 iterations.")
                max_planned_iterations += 50
            else
                Diagnostics.print_error("Hard iteration limit (1000) reached. Stopping simulation.")
                keep_running = false
                is_last_iter = true
            end
        end
        
        do_bin = (save_bin_freq > 0) && (iter % save_bin_freq == 0)
        do_stl = ((save_stl_freq > 0) && (iter % save_stl_freq == 0)) || is_last_iter
        do_vtk = (save_vtk_freq > 0) && (iter % save_vtk_freq == 0)
        
        should_export = do_bin || do_stl || do_vtk || is_last_iter 

        if should_export
            Diagnostics.print_substep("Exporting results...")
            try
                Postprocessing.export_iteration_results(iter, base_name, RESULTS_DIR, nodes, elements, U_full, F_total, bc_indicator, 
                                                        principal_field, vonmises_field, full_stress_voigt, l1_stress_norm_field, 
                                                        principal_max_dir_field, principal_min_dir_field, 
                                                        density, E, geom; 
                                                        iso_threshold=Float32(iso_threshold), 
                                                        current_radius=Float32(current_radius), 
                                                        integral_error=Float32(integral_error),
                                                        config=config, 
                                                        save_bin=do_bin, save_stl=do_stl, save_vtk=do_vtk)
            catch e_export
                Diagnostics.print_error("Post-processing failed at Iter $iter. Logged to crash_report.")
            end
        end
        
        if hard_stop_iter > 0 && iter >= hard_stop_iter; println(">>> HARD STOP: Reached target iteration $hard_stop_iter."); keep_running = false; break; end
        
        if CUDA.functional(); Helpers.clear_gpu_memory(); end
        iter += 1
        GC.gc()
        flush(stdout) 
    end
    Diagnostics.log_status("Finished.")
end

end