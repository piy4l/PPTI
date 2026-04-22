# mini-ppti

`mini-ppti` is a small research prototype for privacy-preserving transformer inference.

The repo currently has two main parts:

- a Python side for symbolic graph construction, partitioning, fusion, and cost estimation
- a C++ side for actual CKKS/OpenFHE execution and microbenchmarks

The current C++ implementation is CPU-only. It uses OpenFHE and OpenMP. There is no CUDA or GPU execution path in this repo.

## What This Repo Can Do

Today the repo supports:

- HE primitive microbenchmarks such as encrypt/decrypt, add, multiply, and rotate
- a toy HE-only transformer-like block
- a tiny BERT-like fixed-shape block
- a tiny token-input text classifier with plaintext vs HE comparison and trace modes
- a baseline benchmark harness for `matmul`, `layernorm`, and `argmax`
- CSV-based comparison against official NEXUS benchmark outputs

What it does not support yet:

- real MPC execution
- full BERT-base or LLM inference
- GPU execution
- exact NEXUS reproduction inside this codebase

## Repo Layout

- [`cpp`](/Users/souvik/Desktop/PPTI/mini-ppti/cpp): OpenFHE-based C++ runtime and benchmarks
- [`python`](/Users/souvik/Desktop/PPTI/mini-ppti/python): graph model, partitioning, symbolic fusion, measured HE cost bridge
- [`scripts`](/Users/souvik/Desktop/PPTI/mini-ppti/scripts): CSV comparison and benchmark-suite drivers
- [`results`](/Users/souvik/Desktop/PPTI/mini-ppti/results): sample/template CSV files for benchmark comparisons
- [`configs`](/Users/souvik/Desktop/PPTI/mini-ppti/configs): placeholder config files, currently not used by the runtime

## Prerequisites

You need:

- CMake
- a C++17 compiler
- OpenFHE installed locally
- OpenMP
- Python 3 for the helper scripts

The current CMake file expects OpenFHE and OpenMP in local install paths referenced by [`cpp/CMakeLists.txt`](/Users/souvik/Desktop/PPTI/mini-ppti/cpp/CMakeLists.txt). If your machine uses different install paths, update that file first.

## Build

Build the C++ binary:

```bash
cd /Users/souvik/Desktop/PPTI/mini-ppti/cpp
cmake -S . -B build
cmake --build build
```

Show all available ops:

```bash
./build/mini_ppti --help
```

## Quick Start

Run a simple HE primitive benchmark:

```bash
cd /Users/souvik/Desktop/PPTI/mini-ppti/cpp
./build/mini_ppti --op mul_ct_ct --n 16 --trials 10 --warmup 3
```

Run the tiny token-input classifier:

```bash
./build/mini_ppti --op tiny_text_classifier --n 16 --tokens 1,2,3,4 --trials 1 --warmup 0
```

Trace the same pipeline stage by stage:

```bash
./build/mini_ppti --op trace_tiny_text_classifier --n 16 --tokens 1,2,3,4
```

## Important Runtime Constraints

- `tiny_bert_block`, `compare_tiny_bert_block`, and `trace_tiny_bert_block` require `--n 16`
- `tiny_text_classifier`, `compare_tiny_text_classifier`, and `trace_tiny_text_classifier` require `--n 16`
- `--tokens` must contain exactly 4 token IDs
- valid token IDs for the tiny text classifier are currently `0` through `7`
- the `argmax` and text-model paths are much slower than the smallest primitive benchmarks

## Main C++ Commands

### Primitive HE Benchmarks

```bash
./build/mini_ppti --op encrypt_only --n 16 --trials 10 --warmup 3
./build/mini_ppti --op decrypt_only --n 16 --trials 10 --warmup 3
./build/mini_ppti --op encrypt_decrypt --n 16 --trials 10 --warmup 3
./build/mini_ppti --op add_plain --n 16 --trials 10 --warmup 3
./build/mini_ppti --op add_ct_ct --n 16 --trials 10 --warmup 3
./build/mini_ppti --op mul_plain --n 16 --trials 10 --warmup 3
./build/mini_ppti --op mul_ct_ct --n 16 --trials 10 --warmup 3
./build/mini_ppti --op rotate --n 16 --steps 1 --trials 10 --warmup 3
```

### Toy Transformer-Like Block

```bash
./build/mini_ppti --op toy_transformer_block --n 8 --trials 2 --warmup 1
./build/mini_ppti --op compare_toy_transformer_block --n 8 --trials 1 --warmup 0
```

### Tiny BERT-Like Fixed-Shape Block

```bash
./build/mini_ppti --op tiny_bert_block --n 16 --trials 1 --warmup 0
./build/mini_ppti --op compare_tiny_bert_block --n 16 --trials 1 --warmup 0
./build/mini_ppti --op trace_tiny_bert_block --n 16
```

### Tiny Token-Input Text Classifier

```bash
./build/mini_ppti --op tiny_text_classifier --n 16 --tokens 1,2,3,4 --trials 1 --warmup 0
./build/mini_ppti --op compare_tiny_text_classifier --n 16 --tokens 1,2,3,4 --trials 1 --warmup 0
./build/mini_ppti --op trace_tiny_text_classifier --n 16 --tokens 1,2,3,4
```

### Baseline Benchmark Kernels

These are the current local comparison kernels used against official NEXUS results.

```bash
./build/mini_ppti --op bench_matmul --impl baseline --n 16 --trials 1 --warmup 0 --csv-only
./build/mini_ppti --op bench_layernorm --impl baseline --n 16 --trials 1 --warmup 0 --csv-only
./build/mini_ppti --op bench_argmax --impl baseline --n 16 --trials 1 --warmup 0 --csv-only
```

The repo also supports `--impl optimized`, but if you are doing an apples-to-apples comparison against the official NEXUS repo, the intended use here is primarily the local `baseline` path.

## Run With Official NEXUS Inputs

The binary can now consume the official NEXUS `data/input` and `data/calibration` files directly for:

- `nexus_gelu`
- `nexus_layernorm`
- `nexus_softmax`
- `nexus_argmax`
- `nexus_matmul`

Point it at a local NEXUS checkout:

```bash
cd /Users/souvik/Desktop/PPTI/mini-ppti/cpp
./build/mini_ppti --op nexus_gelu \
  --impl baseline \
  --nexus-input-dir /path/to/NEXUS/data/input \
  --nexus-calibration-dir /path/to/NEXUS/data/calibration \
  --trials 1 \
  --warmup 0
```

Or run the small wrapper script from the repo root:

```bash
cd /Users/souvik/Desktop/PPTI/mini-ppti
python3 scripts/run_nexus_like_baseline.py \
  --nexus-root /path/to/NEXUS \
  --ops nexus_gelu,nexus_argmax \
  --impl baseline \
  --trials 1 \
  --warmup 0
```

These commands print NEXUS-style lines such as runtime and mean absolute error. The local kernels are still baseline surrogates, not exact NEXUS implementations.
For `nexus_matmul`, the binary reads the official matrix files but benchmarks a documented 4x4-tiled HE surrogate derived from those matrices and prints `Average error` against the matching plaintext surrogate.

## One-Command Baseline Suite

Run the current local suite and write all rows into one CSV:

```bash
cd /Users/souvik/Desktop/PPTI/mini-ppti
python3 scripts/run_baseline_suite.py \
  --output-csv results/baseline_results.csv \
  --impl baseline \
  --trials 1 \
  --warmup 0
```

This currently runs:

- `bench_matmul`
- `bench_layernorm`
- `bench_argmax`

To run the paper-aligned local suite instead:

```bash
python3 scripts/run_baseline_suite.py \
  --output-csv results/baseline_results_paper_aligned.csv \
  --impl baseline \
  --suite paper-aligned \
  --trials 1 \
  --warmup 0
```

The `paper-aligned` suite currently runs:

- `bench_matmul_qkv`
- `bench_matmul_attn_v`
- `bench_matmul_ffn1`
- `bench_matmul_ffn2`
- `bench_layernorm_r128`
- `bench_argmax_r128`

## Compare Against Official NEXUS Results

Put the official NEXUS measurements into:

```bash
results/nexus_results.csv
```

The expected schema is:

```csv
system,op,shape,impl,avg_ms,notes
```

Templates are provided in:

- [`results/nexus_results_template.csv`](/Users/souvik/Desktop/PPTI/mini-ppti/results/nexus_results_template.csv)
- [`results/baseline_results_template.csv`](/Users/souvik/Desktop/PPTI/mini-ppti/results/baseline_results_template.csv)

Join the official NEXUS CSV and the local baseline CSV:

```bash
cd /Users/souvik/Desktop/PPTI/mini-ppti
python3 scripts/compare_benchmarks.py \
  --nexus results/nexus_results.csv \
  --baseline results/baseline_results.csv \
  --output-csv results/comparison.csv
```

If the local baseline kernels do not yet match the exact paper shapes, you can still compare by broad op family:

```bash
python3 scripts/compare_benchmarks.py \
  --nexus results/nexus_results.csv \
  --baseline results/baseline_results.csv \
  --match-mode family \
  --output-csv results/comparison_family.csv
```

## One-Command Head-To-Head Run

If you have a local official NEXUS checkout, this repo now provides a single-command CPU comparison driver.

It will:

- switch the official NEXUS CPU target op
- rebuild the official NEXUS CPU binary
- run the official NEXUS benchmark
- run the local baseline on the same official input/calibration files
- print one joined comparison table

Command:

```bash
cd /Users/souvik/Desktop/PPTI/mini-ppti
python3 scripts/run_head_to_head_comparison.py \
  --nexus-root /path/to/NEXUS \
  --ops matmul,argmax,gelu,layernorm,softmax \
  --impl baseline \
  --trials 1 \
  --warmup 0 \
  --output-csv results/head_to_head.csv
```

Important notes:

- this uses the official NEXUS CPU implementation, not CUDA
- the official NEXUS CPU binary is selected by patching `src/main.cpp` temporarily and rebuilding
- the script restores `src/main.cpp` after it finishes
- the local baseline still uses documented surrogate kernels, especially for matmul

Useful flags:

- `--nexus-build-mode official`
  - use the real official NEXUS CPU build
- `--nexus-build-mode nexus-source-local`
  - use the generated local fallback runner from official NEXUS source files
- `--skipped-csv results/head_to_head_skipped.csv`
  - write skipped ops and failure reasons to a CSV

Recommended supercomputer command:

```bash
cd /Users/souvik/Desktop/PPTI/mini-ppti
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

If the official NEXUS build needs explicit CMake hints:

```bash
python3 scripts/run_head_to_head_comparison.py \
  --nexus-root /path/to/NEXUS \
  --nexus-seal-dir /path/to/SEALConfig.cmake/dir \
  --nexus-cmake-prefix-path /path/to/prefix \
  --ops matmul,argmax,gelu,layernorm,softmax \
  --impl baseline \
  --trials 1 \
  --warmup 0 \
  --nexus-build-mode official \
  --output-csv results/head_to_head.csv \
  --skipped-csv results/head_to_head_skipped.csv
```

Local status before the supercomputer move:

- local primary comparison succeeded for `gelu`, `softmax`, and `matmul`
- output file:
  - [`results/head_to_head_local_primary.csv`](/Users/souvik/Desktop/PPTI/mini-ppti/results/head_to_head_local_primary.csv)
- those local runs used `--nexus-build-mode nexus-source-local`
- they are sanity-check runs, not the final benchmark record

For the next Codex session, start with:

- [`NEXT_CODEX.md`](/Users/souvik/Desktop/PPTI/mini-ppti/NEXT_CODEX.md)

The script prints a markdown table with:

- `op`
- `shape`
- `nexus_ms`
- `baseline_ms`
- `slowdown_baseline_vs_nexus`

Template rows with `0.0` runtime are treated as placeholders and skipped.

`--match-mode family` is a semantic comparison only. It is useful when the local baseline kernels are not yet exact BERT-scale reproductions of the paper workloads.

## Python Symbolic Workflow

The Python side is useful for graph partitioning and measured HE cost estimation.

Run the transformer-like symbolic graph:

```bash
cd /Users/souvik/Desktop/PPTI/mini-ppti/python
python3 run_experiment.py --graph transformer_block --fusion
```

Run the HE bridge test:

```bash
python3 test_he_bridge.py
```

## Reproducibility Notes

If someone else needs to reproduce your results, they should record:

- the git commit hash of this repo
- the git commit hash of the official NEXUS repo
- CPU model
- thread count
- OpenFHE version
- compiler and version
- whether NEXUS was run in CPU or GPU mode
- all benchmark flags used here, especially:
  - `--op`
  - `--impl`
  - `--n`
  - `--trials`
  - `--warmup`

## Where To Look First

If you are editing the runtime:

- [`cpp/src/main.cpp`](/Users/souvik/Desktop/PPTI/mini-ppti/cpp/src/main.cpp): CLI, benchmark harness, CSV append path
- [`cpp/src/ckks_runner.cpp`](/Users/souvik/Desktop/PPTI/mini-ppti/cpp/src/ckks_runner.cpp): actual HE kernels and toy model implementations

If you are editing the symbolic planner:

- [`python/run_experiment.py`](/Users/souvik/Desktop/PPTI/mini-ppti/python/run_experiment.py)
- [`python/cost_model.py`](/Users/souvik/Desktop/PPTI/mini-ppti/python/cost_model.py)
- [`python/fusion.py`](/Users/souvik/Desktop/PPTI/mini-ppti/python/fusion.py)
