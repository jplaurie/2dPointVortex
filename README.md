# PointVortex

PointVortex is a dependency-light C++20 solver for two-dimensional point-vortex dynamics. It
supports the infinite plane, a square periodic box, and a circular disk; CPU/OpenMP, MPI, and
NVIDIA CUDA executables use the same input format, integrators, output files, and checkpoints.

The solver uses direct `O(N^2)` velocity sums. It is a clear numerical reference and a practical
tool for small-to-medium simulations; it is not a tree-code or FMM implementation.

## Quick start

The CPU example needs only CMake 3.20+ and a C++20 compiler. From the repository root:

```bash
cmake -S . -B build/cpu -DCMAKE_BUILD_TYPE=Release \
  -DPOINT_VORTEX_MPI=OFF -DPOINT_VORTEX_CUDA=OFF
cmake --build build/cpu --parallel
ctest --test-dir build/cpu --output-on-failure
./build/cpu/point_vortex_cpu examples/quickstart.params
```

This advances a two-vortex infinite-plane case to time `0.1`. It writes:

| File | Contents |
|---|---|
| `runs/quickstart/trajectory.csv` | Saved positions, circulations, and velocities |
| `runs/quickstart/diagnostics.csv` | Invariants and conservation drift |
| `runs/quickstart/checkpoints/` | Restart checkpoints |

Every simulation has one managed run directory. Missing directories are created automatically.
Existing solver output is protected; use a new `runDirectory` for another experiment, or set
`overwriteRun true` to replace the managed output in that directory.

[`params.txt`](params.txt) is a larger periodic example with 400 vortices, adaptive integration,
and dipole removal/reinjection.

## Features

- Three geometries: `infinite`, `periodic`, and `disk`
- Fixed-step classical RK4 and adaptive Dormand-Prince 5(4) integration
- CPU serial/OpenMP, MPI, and CUDA velocity backends
- Text initial conditions, geometry-aware generator, CSV output, and restart checkpoints
- Optional close dipole removal and reinjection in periodic and disk domains
- CMake and Make builds, numerical tests, and analysis/movie tools

## Repository layout

```text
.
├── src/                  Solver, kernels, integrators, I/O, and backend implementations
├── initial_conditions/   Initial-condition generator and its guide
├── examples/             Small parameter files, including the quick start
├── tests/                C++ unit/audit and Python integration/backend tests
├── scripts/
│   ├── vortices.ipynb    Vortex-configuration plotting notebook
│   ├── diagnostics.ipynb Invariants, drift, and dipole-event notebook
│   ├── movie_vortices.py Vortex-configuration MP4/GIF renderer
│   ├── point_vortex_plotting.py  Shared readers and plotting helpers
│   ├── analysis/         Jupyter notebook for diagnostics and configuration figures
│   └── movie/            Legacy streaming CSV-to-MP4 renderer
├── runs/                 Generated managed run output (ignored by Git)
├── CMakeLists.txt        Primary cross-platform build configuration
├── Makefile              Lightweight alternative build workflow
└── params.txt            Full periodic-run example
```

### `src/` at a glance

| Area | Files | Responsibility |
|---|---|---|
| Program driver | `main.cpp`, `print.cpp` | Loads a run, schedules output, and reports diagnostics |
| Model and diagnostics | `vortex.h`, `compute.cpp/.h` | Vortex storage, velocity kernels, Hamiltonian, and invariants |
| Time integration | `timestep.cpp/.h` | RK4 and adaptive DOPRI5 stepping |
| Execution backends | `backend_cpu.cpp`, `backend_mpi.cpp`, `backend_cuda.cu`, `backend_common.cpp`, `backend.h` | CPU/OpenMP, MPI, CUDA, and shared backend interface |
| Configuration and input | `params.h`, `read.cpp/.h` | Parameter parsing, validation, and initial-condition loading |
| Events and restart | `dipole.cpp/.h`, `checkpoint.cpp/.h` | Dipole handling and versioned checkpoint I/O |
| Benchmark | `benchmark.cpp` | Standalone infinite-plane kernel benchmark |

The velocity kernel is deliberately separate from the timestepper, so a new geometry or faster
kernel can be added without rewriting the integrators.

## Build

### CMake (recommended)

The basic build attempts optional MPI and CUDA targets when their toolchains are installed;
otherwise it still builds the CPU executable.

```bash
cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release
cmake --build build/release --parallel
ctest --test-dir build/release --output-on-failure
```

Use a CPU-only build when MPI or CUDA should not be detected:

```bash
cmake -S . -B build/cpu -DCMAKE_BUILD_TYPE=Release \
  -DPOINT_VORTEX_MPI=OFF -DPOINT_VORTEX_CUDA=OFF
```

Useful configuration options:

| Option | Default | Effect |
|---|---:|---|
| `POINT_VORTEX_OPENMP` | `ON` | Enable OpenMP when the compiler supports it |
| `POINT_VORTEX_MPI` | `ON` | Build `point_vortex_mpi` when MPI is found |
| `POINT_VORTEX_CUDA` | `ON` | Build `point_vortex_cuda` when CUDA is found |
| `POINT_VORTEX_CUDA_ARCHITECTURES` | empty | Optional CUDA target, e.g. `-DPOINT_VORTEX_CUDA_ARCHITECTURES=89` |

### Make

```bash
make                 # CPU/OpenMP executable and initial-condition generator
make test            # C++ numerical tests
make test-output     # Python output/restart integration tests
make mpi             # MPI executable
make cuda            # CUDA executable
make benchmark        # Kernel benchmark
```

Make outputs are under `build/make/`; CMake outputs are under the selected build directory.
They are alternative ways to build the same source tree.

## Run a simulation

All backends receive a parameter file as their optional first argument; with no argument, they
use `params.txt`.

```bash
./build/release/point_vortex_cpu run.params
mpirun -n 4 ./build/release/point_vortex_mpi run.params
./build/release/point_vortex_cuda run.params
```

Start with the CPU backend when checking a new input. The CPU executable uses OpenMP when it was
compiled with support and the population is sufficiently large. Set `numThreads` in the parameter
file, or use `OMP_NUM_THREADS`; a positive `numThreads` takes precedence.

MPI distributes target-vortex calculations across ranks while retaining the complete source state
on each rank. Only rank zero writes output. In a hybrid MPI/OpenMP run, choose ranks × threads to
fit the available CPU cores:

```bash
OMP_NUM_THREADS=8 mpirun -n 2 ./build/release/point_vortex_mpi run.params
```

The CUDA backend requires an NVIDIA GPU, driver, and CUDA toolkit. It keeps vortex state and
RK4/DOPRI5 stages on the GPU between output events, avoiding per-stage host/device transfers.
The host synchronizes state only for output, diagnostics, checkpoints, and enabled dipole
processing. This uses additional GPU memory for the integration-stage buffers.

## Parameter files

Each non-empty line is `key value`; `#` starts a comment. Keys are case-sensitive. Paths cannot
contain whitespace. Invalid or unknown settings stop the run rather than being ignored.

```text
# Minimal two-vortex run
N 2
boundaryCondition infinite
integrator rk4
timeStep 0.001
endTime 0.1
outputTime 0.02
runDirectory runs/my-two-vortex-case
```

Most-used settings:

| Setting | Values / default | Purpose |
|---|---|---|
| `N` | `100` | Built-in initial population; ignored for file/checkpoint input |
| `boundaryCondition` | `infinite` | `infinite`, `periodic`, or `disk` |
| `integrator` | `dopri5` | `rk4` or adaptive `dopri5` |
| `timeStep`, `endTime` | `0.001`, `1.0` | Initial/fixed step and final simulation time |
| `outputTime` | `0.1` | Trajectory interval; final state is always saved |
| `diagnosticsTime`, `checkpointTime` | `outputTime` | Optional independent output intervals |
| `coreRadius` | `0.0` | Infinite-plane regularization radius |
| `numThreads` | `0` | OpenMP thread count; zero defers to the runtime |
| `initialConditionFile` | unset | File with `x y circulation` rows |
| `restartFile` | unset | Checkpoint to restore; overrides the initial condition |
| `runDirectory` | `runs/default` | Self-contained output root for this simulation |
| `overwriteRun` | `false` | Replace this directory's managed solver output |

For a periodic box, set `boxLengthX`, `boxLengthY` (currently equal), and optionally
`periodicImageLayers`; total circulation must be zero. For a disk, set `diskRadius`; every vortex
must remain strictly inside it. `absoluteTolerance`, `relativeTolerance`, `minimumTimeStep`, and
`maximumTimeStep` control adaptive DOPRI5. See the commented
[`params.txt`](params.txt) for every supported setting, including dipole removal and reinjection.

The model is the standard point-vortex equation. In the infinite plane,

```text
dx_i/dt = -1/(2π) Σ[j≠i] Γ_j (y_i-y_j) / (r_ij² + ε²)
dy_i/dt =  1/(2π) Σ[j≠i] Γ_j (x_i-x_j) / (r_ij² + ε²)
```

where `Γ` is circulation and `ε` is `coreRadius`. The periodic implementation follows the
[Weiss–McWilliams square-torus construction](https://atoc.colorado.edu/~jweiss/website/publications/WeissMcWilliams1991.pdf);
the disk uses circle-theorem image vortices.

## Initial conditions

Provide a plain text file with one `x y circulation` row per vortex (whitespace or commas are
accepted), then set `initialConditionFile`:

```text
# initial.dat
-1.0, 0.0,  1.0
 1.0, 0.0, -1.0
```

```text
initialConditionFile initial.dat
```

Or build and use the generator:

```bash
cmake --build build/release --target point_vortex_initial --parallel
./build/release/point_vortex_initial \
  --geometry periodic --case random --count 400 --seed 20261376 \
  --box-length 2 --min-separation 0.01 --output runs/periodic_n400/initial_n400.dat
```

The generator supports `single`, `pair`, `dipole`, `ring`, and `random` cases, records geometry
metadata, and refuses incompatible solver settings. Full options and examples are in
[`initial_conditions/README.md`](initial_conditions/README.md).

## Output, run records, and restarting

Every simulation writes to a self-contained run directory. `runDirectory` defaults to
`runs/default`; give each experiment a descriptive directory:

```text
runDirectory runs/periodic_n400
```

The solver creates this fixed layout, which keeps each experiment's outputs together.

```text
runs/periodic_n400/
├── trajectory.csv
├── diagnostics.csv
├── checkpoints/
├── resolved_parameters.txt
└── segments/
    └── segment_00000001/resolved_parameters.txt
```

`resolved_parameters.txt` records the validated settings, resolved output paths, selected
backend, and available runtime details (OpenMP threads, MPI ranks, or CUDA device). Every fresh
run, restart, or branch gets a new numbered segment record, retaining the provenance of the
invocation even when the top-level record is updated. A run directory that already contains solver
output is rejected by default. Set `overwriteRun true` only when intentionally replacing its
trajectory, diagnostics, checkpoints, and provenance records; unrelated files in that directory
are not removed.

Every backend writes the same portable formats:

| Output | Location | Notes |
|---|---|---|
| Trajectory | `runDirectory/trajectory.csv` | `time,frame,index,x,y,circulation,u,v` rows |
| Diagnostics | `runDirectory/diagnostics.csv` | Invariants, drift, and dipole-event counts |
| Checkpoints | `runDirectory/checkpoints/checkpoint_*.dat` | Versioned restart state |

Trajectory, diagnostics, and checkpoint intervals are simulation time, not wall-clock time.
The solver always saves the initial and final states.

To branch from a checkpoint, create a new parameter file with a new run directory:

```text
restartFile runs/periodic_n400/checkpoints/checkpoint_00000005.dat
endTime 2.0
runDirectory runs/periodic_n400_branch
```

The geometry, integrator, core radius, and dipole settings must match the checkpoint. The
restart begins new CSV files; it does not append to the source trajectory. `frame` is a
monotonically increasing output-event identifier stored in trajectory, diagnostics, and
checkpoints. It lets analysis join streams reliably when their independent schedules coincide;
gaps in an individual CSV mean that stream was not scheduled at that event.

## Analysis and movies

The plotting notebooks read a complete run directory, infer its geometry and box from
`resolved_parameters.txt`, and write figures beneath that run by default. Open the notebooks and
edit their clearly marked **Configuration** cells:

```bash
jupyter lab scripts/vortices.ipynb scripts/diagnostics.ipynb
python3 scripts/movie_vortices.py --run-dir runs/periodic_n400 \
  --output runs/periodic_n400/figures/vortices.mp4
```

The configuration notebook can override the geometry, square or rectangular periodic box, disk
radius, and explicit viewing limits. The movie exposes equivalent command-line options.
Frame selection, GIF/MP4 output, diagnostics ranges, smoothing, dependencies, and more examples
are documented in [`scripts/README.md`](scripts/README.md). The original analysis notebook and
movie entry point remain under `scripts/analysis/` and `scripts/movie/` for compatibility.

## Test and validate changes

Run the CMake test suite after changes to solver code:

```bash
ctest --test-dir build/release --output-on-failure
```

Optional backend comparisons require the corresponding executable/toolchain:

```bash
python3 tests/backend_consistency.py \
  ./build/release/point_vortex_cpu ./build/release/point_vortex_cuda
python3 tests/backend_consistency.py \
  ./build/release/point_vortex_cpu mpirun -n 2 ./build/release/point_vortex_mpi
```

`tests/tests.cpp` covers core numerical behavior; `tests/audit_tests.cpp` targets numerical edge
cases; `tests/output_integration.py` covers output and restart workflows. The analysis and movie
smoke tests can be run with `python3 tests/tooling_tests.py build/release` when their Python
dependencies are installed.

## Limitations

- Velocity evaluation is direct `O(N^2)`; MPI still replicates the source arrays on each rank.
- CUDA keeps velocity evaluation and RK4/DOPRI5 integration on the device; diagnostics and file
  I/O remain host-side, and the stage buffers increase GPU-memory use.
- Periodic dynamics currently requires a square, zero-net-circulation domain.
- Core regularization is available only for the infinite plane.
- Singular encounters, disk-boundary violations, and non-finite states stop the run.
