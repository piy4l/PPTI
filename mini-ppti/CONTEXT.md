# Repo Context For Codex

This file is a compact orientation memo for another Codex session.

## What This Repo Is

`mini-ppti` is a small privacy-preserving transformer inference prototype.

Current emphasis:

- CPU-only CKKS/OpenFHE runtime in C++
- symbolic graph partitioning and cost estimation in Python
- baseline benchmark harness for comparison against the official NEXUS implementation

## Current Comparison Goal

The user has already reproduced official NEXUS results on a supercomputer.

The role of this repo is:

- not to reimplement exact NEXUS optimized kernels
- not to compare against paper numbers only
- but to provide a local CPU baseline / unoptimized comparison target

Current local benchmark kernels used for that comparison:

- `bench_matmul`
- `bench_layernorm`
- `bench_argmax`
- `nexus_gelu`
- `nexus_layernorm`
- `nexus_softmax`
- `nexus_argmax`
- `nexus_matmul`

The intended workflow is:

1. run official NEXUS separately
2. export official results into `results/nexus_results.csv`
3. run the local baseline suite here
4. compare with `scripts/compare_benchmarks.py`

There is now also a direct head-to-head path:

1. point at a NEXUS checkout
2. run [`scripts/run_head_to_head_comparison.py`](/Users/souvik/Desktop/PPTI/mini-ppti/scripts/run_head_to_head_comparison.py)
3. collect one joined CSV plus an optional skipped-op CSV

## What Exists In C++

Main files:

- [`cpp/src/main.cpp`](/Users/souvik/Desktop/PPTI/mini-ppti/cpp/src/main.cpp)
- [`cpp/src/ckks_runner.cpp`](/Users/souvik/Desktop/PPTI/mini-ppti/cpp/src/ckks_runner.cpp)
- [`cpp/include/ckks_runner.h`](/Users/souvik/Desktop/PPTI/mini-ppti/cpp/include/ckks_runner.h)

Implemented features:

- HE primitive benchmarks
- toy transformer-like block
- tiny BERT-like fixed-shape block
- tiny token-input text classifier
- trace modes with per-stage error reporting
- benchmark CSV append path via `--append-csv`

Important current benchmark CLI flags:

- `--op`
- `--impl baseline|optimized`
- `--nexus-input-dir`
- `--nexus-calibration-dir`
- `--input-file`
- `--calibration-file`
- `--n`
- `--trials`
- `--warmup`
- `--append-csv`
- `--csv-only`

## What Exists In Python

Main files:

- [`python/run_experiment.py`](/Users/souvik/Desktop/PPTI/mini-ppti/python/run_experiment.py)
- [`python/cost_model.py`](/Users/souvik/Desktop/PPTI/mini-ppti/python/cost_model.py)
- [`python/fusion.py`](/Users/souvik/Desktop/PPTI/mini-ppti/python/fusion.py)
- [`python/he_bridge.py`](/Users/souvik/Desktop/PPTI/mini-ppti/python/he_bridge.py)

The Python side is symbolic and measured-cost-driven. It is not a full inference engine.

## Important Limitations

- CPU-only
- no GPU path
- no real MPC backend
- no exact BERT-base / LLM inference
- no exact NEXUS kernel reproduction in this repo

## Important Scripts

- [`scripts/run_baseline_suite.py`](/Users/souvik/Desktop/PPTI/mini-ppti/scripts/run_baseline_suite.py)
  Runs the current local baseline suite and writes one CSV.

- [`scripts/compare_benchmarks.py`](/Users/souvik/Desktop/PPTI/mini-ppti/scripts/compare_benchmarks.py)
  Joins official NEXUS CSV rows with local baseline CSV rows.

- [`scripts/run_nexus_like_baseline.py`](/Users/souvik/Desktop/PPTI/mini-ppti/scripts/run_nexus_like_baseline.py)
  Runs the baseline binary directly on official NEXUS `data/input` and `data/calibration` files and prints NEXUS-style runtime/MAE output.

- [`scripts/run_head_to_head_comparison.py`](/Users/souvik/Desktop/PPTI/mini-ppti/scripts/run_head_to_head_comparison.py)
  Single-command CPU comparison driver. It supports:
  - `official` mode for the real official NEXUS CPU build
  - `nexus-source-local` mode for a generated fallback runner from official NEXUS source files

- [`NEXT_CODEX.md`](/Users/souvik/Desktop/PPTI/mini-ppti/NEXT_CODEX.md)
  Minimal handoff file for the next Codex shell. Read this first on the supercomputer.

## Current Recommended Commands

Build:

```bash
cd /Users/souvik/Desktop/PPTI/mini-ppti/cpp
cmake -S . -B build
cmake --build build
```

Run local baseline suite:

```bash
cd /Users/souvik/Desktop/PPTI/mini-ppti
python3 scripts/run_baseline_suite.py \
  --output-csv results/baseline_results.csv \
  --impl baseline \
  --trials 1 \
  --warmup 0
```

Compare against official NEXUS:

```bash
python3 scripts/compare_benchmarks.py \
  --nexus results/nexus_results.csv \
  --baseline results/baseline_results.csv \
  --output-csv results/comparison.csv
```

Run directly on NEXUS input files:

```bash
python3 scripts/run_nexus_like_baseline.py \
  --nexus-root /path/to/NEXUS \
  --ops nexus_gelu,nexus_argmax \
  --impl baseline \
  --trials 1 \
  --warmup 0
```

Run official NEXUS and baseline in one command:

```bash
python3 scripts/run_head_to_head_comparison.py \
  --nexus-root /path/to/NEXUS \
  --ops matmul,argmax,gelu,layernorm,softmax \
  --impl baseline \
  --trials 1 \
  --warmup 0 \
  --output-csv results/head_to_head.csv
```

Preferred supercomputer command:

```bash
python3 scripts/run_head_to_head_comparison.py \
  --nexus-root /path/to/NEXUS \
  --ops matmul,argmax,gelu,layernorm,softmax \
  --impl baseline \
  --trials 1 \
  --warmup 0 \
  --nexus-build-mode official \
  --output-csv results/head_to_head.csv \
  --skipped-csv results/head_to_head_skipped.csv
```

Local sanity-check result already available:

- [`results/head_to_head_local_primary.csv`](/Users/souvik/Desktop/PPTI/mini-ppti/results/head_to_head_local_primary.csv)

That file is useful only as a workflow check. It used:

- `--nexus-build-mode nexus-source-local`

Trace the end-to-end tiny text classifier:

```bash
cd /Users/souvik/Desktop/PPTI/mini-ppti/cpp
./build/mini_ppti --op trace_tiny_text_classifier --n 16 --tokens 1,2,3,4
```

## Good Next Steps

If continuing the benchmark-comparison work, best next tasks are:

1. expand benchmark kernels to better match NEXUS benchmark categories/shapes
2. standardize CSV schemas further if needed for the supercomputer exports
3. add more benchmark suite drivers if additional kernels are added

Immediate next task on the supercomputer:

1. verify official NEXUS dependencies
2. run the `official` head-to-head command
3. inspect `results/head_to_head.csv` and `results/head_to_head_skipped.csv`

Avoid drifting into:

- large new toy-model work
- local optimized reimplementations that are not needed for the official-vs-baseline comparison

Unless the user asks for that explicitly.
