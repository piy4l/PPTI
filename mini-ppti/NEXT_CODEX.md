# Next Codex Handoff

This file is the shortest path for the next Codex shell.

## Goal

Run apples-to-apples baseline vs NEXUS comparisons, preferably on the supercomputer.

## Current State

- `mini-ppti` baseline binary builds and runs locally.
- The one-command comparison driver is:
  - [`scripts/run_head_to_head_comparison.py`](/Users/souvik/Desktop/PPTI/mini-ppti/scripts/run_head_to_head_comparison.py)
- The driver supports two NEXUS-side modes:
  - `official`
  - `nexus-source-local`
- `official` means the real NEXUS CPU build.
- `nexus-source-local` means a generated minimal runner built from official NEXUS source files.

## What Worked Locally

Local primary comparison succeeded for:

- `gelu`
- `softmax`
- `matmul`

Output file:

- [`results/head_to_head_local_primary.csv`](/Users/souvik/Desktop/PPTI/mini-ppti/results/head_to_head_local_primary.csv)

The mode for those local runs was:

- `nexus-source-local`

Reason:

- this laptop did not have the full official NEXUS CPU dependency chain available
- specifically `NTL` blocked the real official CPU build

## What Did Not Work Reliably Locally

- `argmax`
  - official CPU build depends on `NTL`
- `layernorm`
  - the local baseline `nexus_layernorm` run was killed on this machine

These should be retried on the supercomputer, not debugged on the laptop unless necessary.

## Supercomputer Target Workflow

First objective on the supercomputer:

1. build `mini-ppti`
2. build official NEXUS CPU and/or CUDA with all dependencies available
3. run [`scripts/run_head_to_head_comparison.py`](/Users/souvik/Desktop/PPTI/mini-ppti/scripts/run_head_to_head_comparison.py) in `official` mode
4. produce one comparison CSV and one skipped-op CSV

## Recommended Command On The Supercomputer

From repo root:

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

If official NEXUS needs explicit SEAL/CMake hints:

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

## Dependencies To Verify On The Supercomputer

For `mini-ppti`:

- OpenFHE
- OpenMP

For official NEXUS CPU:

- SEAL 4.1
- NTL
- GMP
- pthread
- any required include/library paths visible to CMake

For official NEXUS CUDA, if used later:

- CUDA toolchain
- whatever their `cuda/README.md` requires

## Important Output Files

- local primary comparison:
  - [`results/head_to_head_local_primary.csv`](/Users/souvik/Desktop/PPTI/mini-ppti/results/head_to_head_local_primary.csv)
- local subset:
  - [`results/head_to_head_local_subset.csv`](/Users/souvik/Desktop/PPTI/mini-ppti/results/head_to_head_local_subset.csv)
- local gelu-only:
  - [`results/head_to_head_local_gelu.csv`](/Users/souvik/Desktop/PPTI/mini-ppti/results/head_to_head_local_gelu.csv)

## Important Caveat

Local numbers from `nexus-source-local` are only a sanity-check and workflow validation.

They are not the final benchmark record.

The final benchmark record should come from the supercomputer with:

- `--nexus-build-mode official`

## First Thing The Next Codex Shell Should Do

1. read this file
2. read [`CONTEXT.md`](/Users/souvik/Desktop/PPTI/mini-ppti/CONTEXT.md)
3. verify dependencies on the supercomputer
4. run the `official` command above
