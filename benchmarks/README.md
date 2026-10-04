# Backend benchmarks

The backend benchmark measures complete fixed-step RK4 updates for the infinite-plane model. Each
step performs four direct all-pairs velocity evaluations. The timed interval starts only after
backend construction, memory allocation, CUDA context creation/state upload, and warm-up steps.
The final CUDA download and checksum are also outside the timed interval.

Configure and build all available benchmark executables:

```bash
cmake -S . -B build/benchmarks -DCMAKE_BUILD_TYPE=Release
cmake --build build/benchmarks --parallel --target \
  point_vortex_backend_benchmark_serial \
  point_vortex_backend_benchmark_cpu \
  point_vortex_backend_benchmark_mpi \
  point_vortex_backend_benchmark_cuda \
  point_vortex_backend_benchmark_cuda_mixed
```

Run five calibrated trials at each problem size and create the README-ready plot:

```bash
python3 benchmarks/run_benchmarks.py --overwrite
python3 benchmarks/plot_benchmarks.py
```

By default CPU/OpenMP uses one thread per detected physical core, MPI uses the same number of
single-threaded ranks, CPU serial uses a library compiled without OpenMP, and both CUDA modes use
one GPU. Override the parallel CPU choices with `--cpu-threads` and `--mpi-ranks`. Raw trials go to
`results.csv`; machine/build metadata and the exact timing exclusions go to `system.json`.

The plot's timing panel reports absolute RK4 step time. Its speedup panel is normalized by the
single-core `CPU serial` backend, which is the conventional baseline for total parallel speedup.

The checked-in plot extends the default sweep to larger populations with three five-second scout
trials:

```bash
python3 benchmarks/run_benchmarks.py --resolutions 16384 32768 65536 --trials 3 \
  --target-seconds 5 --warmup-seconds 1 \
  --output benchmarks/results_large.csv \
  --metadata benchmarks/system_large.json --overwrite
python3 benchmarks/plot_benchmarks.py \
  --input benchmarks/results.csv benchmarks/results_large.csv \
          benchmarks/results_serial_mixed.csv benchmarks/results_serial_mixed_large.csv
```

For the sustained high-resolution check, use one-minute timed trials and a time-calibrated warm-up.
This makes clock settling and thermal effects visible at the most expensive resolution:

```bash
python3 benchmarks/run_benchmarks.py --resolutions 65536 --trials 3 \
  --target-seconds 60 --warmup-seconds 5 \
  --output benchmarks/results_sustained_65536.csv \
  --metadata benchmarks/system_sustained_65536.json --overwrite
```

The repository also retains the corresponding one-minute check at `N=8192` so the short and
sustained measurements can be compared at both ends of the extended range. The serial CPU and
mixed-CUDA trials at that size can be reproduced with:

```bash
python3 benchmarks/run_benchmarks.py --backends serial cuda_mixed \
  --resolutions 8192 --trials 3 --target-seconds 60 --warmup-seconds 5 \
  --output benchmarks/results_serial_mixed_sustained_8192.csv \
  --metadata benchmarks/system_serial_mixed_sustained_8192.json --overwrite
```

Serial CPU and mixed-CUDA sustained results are stored in the matching
`results_serial_mixed_sustained_8192.csv` and `results_serial_mixed_sustained_65536.csv` files.
