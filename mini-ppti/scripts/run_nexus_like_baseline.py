#!/usr/bin/env python3

import argparse
import subprocess
import sys
from pathlib import Path


OPS = [
    "nexus_gelu",
    "nexus_layernorm",
    "nexus_softmax",
    "nexus_argmax",
    "nexus_matmul",
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the local baseline on official NEXUS input/calibration files."
    )
    parser.add_argument(
        "--binary",
        default="cpp/build/mini_ppti",
        help="Path to the mini_ppti binary relative to repo root or absolute path.",
    )
    parser.add_argument(
        "--nexus-root",
        required=True,
        help="Path to a local checkout of the official NEXUS repository.",
    )
    parser.add_argument("--impl", default="baseline", choices=["baseline", "optimized"])
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument(
        "--ops",
        default="all",
        help="Comma-separated subset from nexus_gelu,nexus_layernorm,nexus_softmax,nexus_argmax, or all.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    binary = Path(args.binary)
    if not binary.is_absolute():
        binary = repo_root / binary
    binary = binary.resolve()

    nexus_root = Path(args.nexus_root).resolve()
    input_dir = nexus_root / "data" / "input"
    calibration_dir = nexus_root / "data" / "calibration"

    if not binary.exists():
        raise SystemExit(f"Binary not found: {binary}")
    if not input_dir.exists():
        raise SystemExit(f"NEXUS input dir not found: {input_dir}")
    if not calibration_dir.exists():
        raise SystemExit(f"NEXUS calibration dir not found: {calibration_dir}")

    selected_ops = OPS if args.ops == "all" else [op.strip() for op in args.ops.split(",") if op.strip()]
    unknown_ops = sorted(set(selected_ops) - set(OPS))
    if unknown_ops:
        raise SystemExit(f"Unsupported --ops entries: {', '.join(unknown_ops)}")

    for op in selected_ops:
        cmd = [
            str(binary),
            "--op",
            op,
            "--impl",
            args.impl,
            "--nexus-input-dir",
            str(input_dir),
            "--nexus-calibration-dir",
            str(calibration_dir),
            "--trials",
            str(args.trials),
            "--warmup",
            str(args.warmup),
        ]
        print(f"$ {' '.join(cmd)}", flush=True)
        completed = subprocess.run(cmd, cwd=repo_root)
        if completed.returncode != 0:
            return completed.returncode
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
