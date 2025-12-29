Here is a comprehensive `README.md` file tailored for the HEXA TopOpt codebase.

```markdown
# HEXA TopOpt: High-Performance GPU Topology Optimization

**HEXA TopOpt** is a robust, high-performance Finite Element Analysis (FEM) and Topology Optimization solver written in Julia. It is designed to solve large-scale structural compliance minimization problems with stress constraints, utilizing NVIDIA GPUs (CUDA) for massive parallelization.

The solver features Geometric Multigrid (GMG) preconditioning, Adaptive Mesh Refinement (AMR), and a flexible configuration system that separates physical problem definitions from numerical solver settings.

---

## 🚀 Key Features

* **GPU Acceleration:** Fully vectorized CUDA kernels for element stiffness generation, assembly, and linear solving (CG). Automatic VRAM management and fallback to CPU if necessary.
* **Geometric Multigrid (GMG):** Memory-efficient matrix-free multigrid preconditioner allowing for solves on meshes with millions of elements.
* **Adaptive Mesh Refinement:** Starts with a coarse grid and progressively refines the mesh around active design regions to capture fine details without checking memory limits.
* **Robust Optimization:**
    * Stress-constrained topology optimization.
    * Explicit density filtering (Gaussian-like diffusion).
    * Strict void/solid region enforcement (geometric primitives).
* **Production Ready IO:**
    * **Input:** Modular YAML configuration files.
    * **Output:** VTK (`.vti`) for Paraview, Watertight STL (`.stl`) for manufacturing, and binary checkpoints.
    * **Batch Processing:** Queue multiple simulation cases with override capabilities.

---

## 📦 Installation

### Prerequisites
1.  **Julia:** Version 1.6 or higher.
2.  **Hardware:** NVIDIA GPU (RTX, V100, A100, H100) recommended. CPU mode is available but significantly slower.
3.  **Drivers:** CUDA toolkit compatible with your GPU.

### Setup
1.  Clone the repository:
    ```bash
    git clone [https://github.com/your-username/HEXA_TopOpt.git](https://github.com/your-username/HEXA_TopOpt.git)
    cd HEXA_TopOpt
    ```

2.  Instantiate the environment (this installs required packages like `CUDA.jl`, `LinearAlgebra`, `YAML`, etc.):
    ```bash
    julia --project=. -e 'using Pkg; Pkg.instantiate()'
    ```

---

## 🏃 Usage

The entry point for the software is `Run.jl`. This script handles environment activation, hardware checks, and launches the simulation batch processor.

### Basic Execution
To run the simulations defined in `configs/optimization_cases.yaml`:

```bash
julia --project=. Run.jl

```

### How it Works

1. **Initialization:** `Run.jl` verifies dependencies and checks for a GPU.
2. **Hardware Profiling:** It runs a stress test on the GPU to generate `configs/_machine_limits.jl`, determining the maximum safe element count to prevent Out-Of-Memory (OOM) crashes.
3. **Batch Processing:** It reads the job queue and sequentially executes simulations.

---

## ⚙️ Configuration

The solver uses a split-configuration approach to maximize reusability. Files are located in the `configs/` directory.

### 1. Domain Configuration (`domain_definitions.yaml`)

Defines the "What" — the physical problem.

* **Geometry:** Domain size (`length_x/y/z`) and primitives (`sphere`, `box`) to define non-design spaces (voids or solid supports).
* **Boundary Conditions:** Defined via coordinate ranges (e.g., `location: [0, ':', ':']` fixes the X=0 face).
* **Loads:** Point loads or distributed loads applied to specific locations.
* **Material:** Young's modulus (`E`), Poisson's ratio (`nu`), density, and stress limits.

### 2. Solver Configuration (`solver_settings.yaml`)

Defines the "How" — the numerical approach.

* **Mesh Settings:** Initial resolution and AMR targets (`final_target_of_active_elements`).
* **Optimization:** Iteration limits, filter radius (`minimum_feature_size`), and culling aggressiveness (`max_culling_ratio`).
* **Solver:** Tolerance, max iterations, and preconditioner selection (`multigrid`).
* **Output:** Export frequency for VTK/STL files.

### 3. Batch Orchestrator (`optimization_cases.yaml`)

Combines domains and solvers into specific jobs. You can use **overrides** to tweak parameters without creating new files.

```yaml
batch_queue:
  - job_name: "Bridge_HighRes"
    domain_config: "./configs/bridge_v1.yaml"
    solver_config: "./configs/solver_settings.yaml"
    overrides:
      mesh_settings:
        final_target_of_active_elements: 10_000_000

```

---

## 📂 Project Structure

```text
HEXA_TopOpt/
├── configs/                 # YAML configuration files
│   ├── bridge_v1.yaml       # Example Physics
│   ├── solver_settings.yaml # Example Solver settings
│   └── optimization_cases.yaml # Batch queue
├── src/
│   ├── Main.jl              # Module entry point
│   ├── App/                 # Simulation loop and Hardware profiling
│   ├── Core/                # FEM Elements, Boundaries, and Stress calc
│   ├── Mesh/                # Mesh generation, Pruning, and Refinement
│   ├── Optimization/        # Topology Optimization and Filtering logic
│   ├── Solvers/             # GPU/CPU Linear Solvers (CG, GMG)
│   ├── IO/                  # Config parsing and VTK/STL Exporting
│   └── Utils/               # Diagnostics and Helper functions
├── RESULTS/                 # Generated output (created at runtime)
├── Run.jl                   # Main Launcher script
└── Project.toml             # Julia dependencies

```

---

## 📊 Outputs

Results are saved in the `RESULTS/<job_name>` directory:

* **`*_solution.vti`**: Full 3D volumetric data (Density, Stress, Displacement) viewable in [ParaView](https://www.paraview.org/).
* **`*_isosurface.stl`**: Smoothed, watertight STL mesh ready for 3D printing or CAD import.
* **`*_animation.pvd`**: Time-series file to animate the optimization process in ParaView.
* **`simulation_log.txt`**: Detailed convergence logs (compliance, volume fraction, stability metrics).

---

## 🛠 Troubleshooting

* **UndefVarError: f0**: If you encounter this in `Boundary.jl`, ensure your float literals use correct syntax (e.g., use `1.0f-5`, not mixed notation like `1.0e-5f0`).
* **OOM (Out of Memory):** Delete `configs/_machine_limits.jl` and restart `Run.jl`. The launcher will re-run the VRAM stress test to calibrate for your current hardware.
* **Instability:** If the solution explodes, check your `boundary_conditions`. Ensure your constraints prevent rigid body motion in all 3 axes.

---

## License

This project is released under the MIT License.

```

```
