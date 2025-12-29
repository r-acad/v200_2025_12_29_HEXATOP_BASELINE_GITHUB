# FILE: .\src\App\BatchProcessor.jl
module BatchProcessor

using CUDA
using ..Configuration
using ..Diagnostics
using ..Helpers
using ..SimulationLoop

export process_batch

function process_batch(input_file=nothing, project_root::String="")
    try
        if input_file === nothing
            input_file = joinpath(project_root, "configs", "optimization_cases.yaml")
        end
        
        if !isfile(input_file)
            error("Input file not found: $input_file")
        end

        raw_config = Configuration.load_configuration(input_file)

        if haskey(raw_config, "batch_queue")
            queue = raw_config["batch_queue"]
            Diagnostics.print_banner("BATCH EXECUTION STARTED: $(length(queue)) Runs")
            
            for (i, run_def) in enumerate(queue)
                job_name = get(run_def, "job_name", "Run_$i")
                Diagnostics.print_banner("BATCH RUN $i/$((length(queue))): $job_name", color="\u001b[35m")
                
                domain_file = get(run_def, "domain_config", "")
                solver_file = get(run_def, "solver_config", "")
                overrides = get(run_def, "overrides", Dict())

                if isempty(domain_file) || isempty(solver_file)
                    Diagnostics.print_error("Skipping $job_name: Missing config file paths.")
                    continue
                end

                if !isabspath(domain_file); domain_file = joinpath(project_root, domain_file); end
                if !isabspath(solver_file); solver_file = joinpath(project_root, solver_file); end

                try
                    merged_config = Configuration.load_and_merge_configurations(domain_file, solver_file, overrides)
                    SimulationLoop.run_simulation(merged_config, job_name; project_root=project_root)
                catch e_run
                    Diagnostics.print_error("Job $job_name Failed: $e_run")
                    showerror(stdout, e_run, catch_backtrace())
                end
                
                Diagnostics.print_success("Finished Batch Run: $job_name")
                GC.gc()
                if CUDA.functional(); CUDA.reclaim(); end
            end
            Diagnostics.print_banner("BATCH QUEUE COMPLETE")

        else
            base_job_name = get(raw_config, "job_name", splitext(basename(input_file))[1])
            SimulationLoop.run_simulation(raw_config, base_job_name; project_root=project_root)
        end

    catch e
        if isa(e, InterruptException)
            Diagnostics.print_banner("USER INTERRUPT", color="\u001b[33m")
            println(">>> Simulation stopped by user.")
        else
            Diagnostics.print_banner("FATAL ERROR DETECTED", color="\u001b[31m")
            showerror(stderr, e, catch_backtrace())
        end
        flush(stdout)
    finally
        if CUDA.functional()
            Diagnostics.print_info("Finalizing: Cleaning up GPU Memory...")
            Helpers.clear_gpu_memory()
            Diagnostics.print_success("GPU Memory Released.")
            flush(stdout)
        end
    end
end

end