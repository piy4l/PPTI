#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path


def parse_metadata(value: str) -> dict[str, str]:
    result: dict[str, str] = {}
    if not value:
        return result

    for item in value.split(";"):
        if not item or "=" not in item:
            continue
        key, raw_value = item.split("=", 1)
        result[key.strip()] = raw_value.strip()
    return result


def canonical_op_family(op: str) -> str:
    value = op.strip().lower()

    if value in {"bench_layernorm", "bench_layernorm_runtime_breakdown", "bench_layernorm_r128"}:
        return "layernorm"
    if value in {"bench_argmax", "bench_argmax_r128"}:
        return "argmax"
    if value in {
        "bench_matmul",
        "bench_matmul_qkv",
        "bench_matmul_attn_v",
        "bench_matmul_ffn1",
        "bench_matmul_ffn2",
    }:
        return "matmul"
    if value == "bench_softmax":
        return "softmax"
    if value == "bench_gelu":
        return "gelu"
    return value


def first_present(row: dict[str, str], candidates: list[str]) -> str:
    for key in candidates:
        value = row.get(key, "")
        if value:
            return value
    return ""


def infer_runtime_ms(row: dict[str, str]) -> float:
    direct_ms = first_present(row, ["avg_ms", "runtime_ms", "time_ms", "latency_ms"])
    if direct_ms:
        return float(direct_ms)

    seconds = first_present(row, ["avg_s", "runtime_s", "time_s", "latency_s"])
    if seconds:
        return float(seconds) * 1000.0

    raise ValueError(f"Could not infer runtime from row: {row}")


def normalize_row(row: dict[str, str], default_system: str, default_impl: str) -> dict[str, str | float]:
    metadata = parse_metadata(row.get("metadata", ""))

    op = first_present(row, ["op", "op_name", "benchmark", "kernel", "name"])
    if not op:
        raise ValueError(f"Could not infer op from row: {row}")

    shape = first_present(row, ["shape", "workload", "config"])
    if not shape:
        shape = metadata.get("shape", "")

    system = first_present(row, ["system"])
    if not system:
        system = default_system

    impl = first_present(row, ["impl"])
    if not impl:
        impl = metadata.get("impl", default_impl)

    runtime_ms = infer_runtime_ms(row)

    return {
        "system": system,
        "op": op,
        "shape": shape,
        "impl": impl,
        "runtime_ms": runtime_ms,
    }


def load_rows(path: Path,
              default_system: str,
              default_impl: str,
              match_mode: str) -> dict[tuple[str, str], float]:
    grouped: dict[tuple[str, str], list[float]] = {}

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            row = normalize_row(raw_row, default_system, default_impl)
            if match_mode == "exact":
                key = (str(row["op"]), str(row["shape"]))
            elif match_mode == "family":
                key = (canonical_op_family(str(row["op"])), "")
            else:
                raise ValueError(f"Unsupported match mode: {match_mode}")
            grouped.setdefault(key, []).append(float(row["runtime_ms"]))

    return {
        key: sum(values) / len(values)
        for key, values in grouped.items()
    }


def render_markdown(rows: list[dict[str, str | float]]) -> str:
    lines = [
        "| key | nexus_shape | baseline_shape | nexus_ms | baseline_ms | slowdown_baseline_vs_nexus |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {key} | {nexus_shape} | {baseline_shape} | {nexus_ms:.3f} | {baseline_ms:.3f} | {slowdown:.3f} |".format(
                key=row["key"],
                nexus_shape=row["nexus_shape"] or "-",
                baseline_shape=row["baseline_shape"] or "-",
                nexus_ms=row["nexus_ms"],
                baseline_ms=row["baseline_ms"],
                slowdown=row["slowdown"],
            )
        )
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict[str, str | float]]) -> None:
    fieldnames = ["key", "nexus_shape", "baseline_shape", "nexus_ms", "baseline_ms", "slowdown"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "key": row["key"],
                    "nexus_shape": row["nexus_shape"],
                    "baseline_shape": row["baseline_shape"],
                    "nexus_ms": f"{row['nexus_ms']:.6f}",
                    "baseline_ms": f"{row['baseline_ms']:.6f}",
                    "slowdown": f"{row['slowdown']:.6f}",
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Join official NEXUS benchmark CSVs with local baseline benchmark CSVs."
    )
    parser.add_argument("--nexus", required=True, help="Path to official NEXUS CSV results.")
    parser.add_argument("--baseline", required=True, help="Path to local baseline CSV results.")
    parser.add_argument(
        "--output-csv",
        help="Optional path to write the joined comparison rows as CSV.",
    )
    parser.add_argument(
        "--match-mode",
        default="exact",
        choices=["exact", "family"],
        help="How to join results. 'exact' matches op+shape. 'family' matches broad op families such as matmul/layernorm/argmax.",
    )
    args = parser.parse_args()

    nexus_rows = load_rows(
        Path(args.nexus),
        default_system="nexus_official",
        default_impl="optimized",
        match_mode=args.match_mode,
    )
    baseline_rows = load_rows(
        Path(args.baseline),
        default_system="mini_ppti_baseline",
        default_impl="baseline",
        match_mode=args.match_mode,
    )

    joined_rows: list[dict[str, str | float]] = []
    all_keys = sorted(set(nexus_rows) | set(baseline_rows))
    for key in all_keys:
        if key not in nexus_rows or key not in baseline_rows:
            continue

        key_name, _ = key
        nexus_ms = nexus_rows[key]
        baseline_ms = baseline_rows[key]
        if nexus_ms <= 0.0 or baseline_ms <= 0.0:
            continue
        joined_rows.append(
            {
                "key": key_name,
                "nexus_shape": key[1] if args.match_mode == "exact" else "paper_mixed_family",
                "baseline_shape": key[1] if args.match_mode == "exact" else "local_mixed_family",
                "nexus_ms": nexus_ms,
                "baseline_ms": baseline_ms,
                "slowdown": baseline_ms / nexus_ms,
            }
        )

    print(render_markdown(joined_rows))

    if args.output_csv:
        write_csv(Path(args.output_csv), joined_rows)


if __name__ == "__main__":
    main()
