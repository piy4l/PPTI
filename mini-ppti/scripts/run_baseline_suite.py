#!/usr/bin/env python3

import argparse
import csv
import subprocess
import tempfile
from pathlib import Path


BENCHMARKS = [
    ("bench_matmul", 16),
    ("bench_layernorm", 16),
    ("bench_argmax", 16),
]

PAPER_ALIGNED_BENCHMARKS = [
    ("bench_matmul_qkv", 128),
    ("bench_matmul_attn_v", 128),
    ("bench_matmul_ffn1", 128),
    ("bench_matmul_ffn2", 128),
    ("bench_layernorm_r128", 128),
    ("bench_argmax_r128", 128),
]


def run_command(cmd: list[str], cwd: Path) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the local baseline benchmark suite and append results to a CSV."
    )
    parser.add_argument(
        "--binary",
        default="/Users/souvik/Desktop/PPTI/mini-ppti/cpp/build/mini_ppti",
        help="Path to the mini_ppti benchmark binary.",
    )
    parser.add_argument(
        "--output-csv",
        default="/Users/souvik/Desktop/PPTI/mini-ppti/results/baseline_results.csv",
        help="Path to the CSV file to append results into.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=1,
        help="Number of benchmark trials per kernel.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Number of warmup runs per kernel.",
    )
    parser.add_argument(
        "--impl",
        default="baseline",
        choices=["baseline", "optimized"],
        help="Which implementation mode to run.",
    )
    parser.add_argument(
        "--suite",
        default="current",
        choices=["current", "paper-aligned"],
        help="Which benchmark set to run.",
    )
    args = parser.parse_args()

    binary = Path(args.binary).resolve()
    if not binary.exists():
        raise FileNotFoundError(f"Benchmark binary not found: {binary}")

    cwd = binary.parent
    output_csv = Path(args.output_csv).resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if output_csv.exists():
        output_csv.unlink()

    benchmarks = BENCHMARKS if args.suite == "current" else PAPER_ALIGNED_BENCHMARKS

    written_rows: list[dict[str, str]] = []
    for op, n in benchmarks:
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_csv = Path(tmpdir) / f"{op}.csv"
            cmd = [
                str(binary),
                "--op",
                op,
                "--impl",
                args.impl,
                "--n",
                str(n),
                "--trials",
                str(args.trials),
                "--warmup",
                str(args.warmup),
                "--append-csv",
                str(temp_csv),
                "--csv-only",
            ]
            run_command(cmd, cwd)

            with temp_csv.open(newline="") as handle:
                reader = csv.DictReader(handle)
                written_rows.extend(reader)

    with output_csv.open("w", newline="") as handle:
        fieldnames = ["system", "op", "shape", "impl", "avg_ms", "notes"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in written_rows:
            writer.writerow(row)

    print(f"Wrote benchmark rows to {output_csv}")


if __name__ == "__main__":
    main()
