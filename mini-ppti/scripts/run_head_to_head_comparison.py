#!/usr/bin/env python3

import argparse
import csv
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


OPS = {
    "matmul": {
        "nexus_idx": 0,
        "nexus_label": "MatMul",
        "baseline_op": "nexus_matmul",
        "baseline_label": "MatMul",
        "metric_label": "Average error",
        "fallback_supported": True,
    },
    "argmax": {
        "nexus_idx": 1,
        "nexus_label": "Argmax",
        "baseline_op": "nexus_argmax",
        "baseline_label": "Argmax",
        "metric_label": "Mean Absolute Error",
        "fallback_supported": False,
    },
    "gelu": {
        "nexus_idx": 2,
        "nexus_label": "GELU",
        "baseline_op": "nexus_gelu",
        "baseline_label": "GELU",
        "metric_label": "Mean Absolute Error",
        "fallback_supported": True,
    },
    "layernorm": {
        "nexus_idx": 3,
        "nexus_label": "LayerNorm",
        "baseline_op": "nexus_layernorm",
        "baseline_label": "LayerNorm",
        "metric_label": "Mean Absolute Error",
        "fallback_supported": True,
    },
    "softmax": {
        "nexus_idx": 4,
        "nexus_label": "Softmax",
        "baseline_op": "nexus_softmax",
        "baseline_label": "Softmax",
        "metric_label": "Mean Absolute Error",
        "fallback_supported": True,
    },
}


@dataclass
class Result:
    runtime_ms: float
    metric_name: str
    metric_value: float
    raw_output: str


def run_checked(cmd: list[str], cwd: Path) -> str:
    completed = subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {' '.join(cmd)}\n{completed.stdout}"
        )
    return completed.stdout


def patch_nexus_target(main_cpp: Path, target_idx: int) -> str:
    original = main_cpp.read_text()
    patched, count = re.subn(
        r"int TEST_TARGET_IDX = \d+;",
        f"int TEST_TARGET_IDX = {target_idx};",
        original,
        count=1,
    )
    if count != 1:
        raise RuntimeError("Could not patch TEST_TARGET_IDX in official NEXUS main.cpp")
    main_cpp.write_text(patched)
    return original


def parse_runtime(output: str, label: str) -> float:
    pattern = re.compile(
        rf"\[{re.escape(label)}\].*?takes:\s*([0-9]+(?:\.[0-9]+)?)\s*milliseconds",
        re.I,
    )
    match = pattern.search(output)
    if not match:
        raise RuntimeError(f"Could not parse runtime for {label}\n{output}")
    return float(match.group(1))


def parse_metric(output: str, metric_name: str) -> float:
    pattern = re.compile(rf"{re.escape(metric_name)}:\s*([0-9]+(?:\.[0-9]+)?)", re.I)
    match = pattern.search(output)
    if not match:
        raise RuntimeError(f"Could not parse {metric_name}\n{output}")
    return float(match.group(1))


def parse_result(output: str, label: str, metric_name: str) -> Result:
    return Result(
        runtime_ms=parse_runtime(output, label),
        metric_name=metric_name,
        metric_value=parse_metric(output, metric_name),
        raw_output=output,
    )


def print_table(rows: list[dict[str, object]]) -> None:
    headers = [
        "op",
        "nexus_ms",
        "baseline_ms",
        "baseline_vs_nexus",
        "nexus_metric",
        "baseline_metric",
        "mode",
    ]
    widths = {h: len(h) for h in headers}
    rendered = []
    for row in rows:
        rendered_row = {
            "op": str(row["op"]),
            "nexus_ms": f"{row['nexus_ms']:.6f}",
            "baseline_ms": f"{row['baseline_ms']:.6f}",
            "baseline_vs_nexus": f"{row['baseline_vs_nexus']:.6f}",
            "nexus_metric": f"{row['nexus_metric_name']}={row['nexus_metric_value']:.6f}",
            "baseline_metric": f"{row['baseline_metric_name']}={row['baseline_metric_value']:.6f}",
            "mode": str(row["nexus_mode"]),
        }
        rendered.append(rendered_row)
        for key, value in rendered_row.items():
            widths[key] = max(widths[key], len(value))

    print("  ".join(h.ljust(widths[h]) for h in headers))
    print("  ".join("-" * widths[h] for h in headers))
    for row in rendered:
        print("  ".join(row[h].ljust(widths[h]) for h in headers))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "op",
                "nexus_ms",
                "baseline_ms",
                "baseline_vs_nexus",
                "nexus_metric_name",
                "nexus_metric_value",
                "baseline_metric_name",
                "baseline_metric_value",
                "nexus_mode",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["op"],
                    f"{row['nexus_ms']:.6f}",
                    f"{row['baseline_ms']:.6f}",
                    f"{row['baseline_vs_nexus']:.6f}",
                    row["nexus_metric_name"],
                    f"{row['nexus_metric_value']:.6f}",
                    row["baseline_metric_name"],
                    f"{row['baseline_metric_value']:.6f}",
                    row["nexus_mode"],
                ]
            )


def write_failures_csv(path: Path, failures: list[tuple[str, str]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["op", "reason"])
        for op, reason in failures:
            writer.writerow([op, reason])


def make_configure_cmd(args: argparse.Namespace) -> list[str]:
    cmd = ["cmake", ".."]
    if args.nexus_seal_dir:
        cmd.append(f"-DSEAL_DIR={args.nexus_seal_dir}")
    if args.nexus_cmake_prefix_path:
        cmd.append(f"-DCMAKE_PREFIX_PATH={args.nexus_cmake_prefix_path}")
    return cmd


def official_nexus_run(
    op: str,
    meta: dict[str, object],
    args: argparse.Namespace,
    nexus_root: Path,
) -> tuple[Result, str]:
    nexus_main = nexus_root / "src" / "main.cpp"
    nexus_build = nexus_root / "build"
    nexus_binary = nexus_build / "bin" / "main"

    if not nexus_build.exists():
        nexus_build.mkdir(parents=True, exist_ok=True)

    run_checked(make_configure_cmd(args), cwd=nexus_build)
    original_main = nexus_main.read_text()
    try:
        patch_nexus_target(nexus_main, int(meta["nexus_idx"]))
        run_checked(["cmake", "--build", "."], cwd=nexus_build)
        output = run_checked([str(nexus_binary)], cwd=nexus_build)
        return parse_result(output, str(meta["nexus_label"]), str(meta["metric_label"])), "official"
    finally:
        nexus_main.write_text(original_main)


def fallback_source_filename(op: str) -> str:
    return {
        "gelu": "gelu.cpp",
        "softmax": "softmax.cpp",
        "layernorm": "layer_norm.cpp",
        "matmul": "matrix_mul.cpp",
    }[op]


def fallback_main_source(op: str, nexus_root: Path) -> str:
    input_dir = nexus_root / "data" / "input"
    cal_dir = nexus_root / "data" / "calibration"
    common_prefix = f"""
#include <seal/seal.h>
#include <seal/util/uintarith.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "{op if op != 'layernorm' else 'layer_norm'}.h"
#include "ckks_evaluator.h"
using namespace std;
using namespace seal;
using namespace seal::util;
using namespace std::chrono;

static vector<double> read_flat(const string &path) {{
  ifstream in(path);
  double num;
  vector<double> out;
  while (in >> num) {{
    out.push_back(num);
  }}
  return out;
}}
"""
    if op == "gelu":
        return common_prefix + f"""
int main() {{
  long logN = 16;
  size_t poly_modulus_degree = 1 << logN;
  vector<int> COEFF_MODULI = {{58, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 58}};
  double SCALE = pow(2.0, 40);

  EncryptionParameters parms(scheme_type::ckks);
  parms.set_poly_modulus_degree(poly_modulus_degree);
  parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, COEFF_MODULI));
  SEALContext context(parms, true, sec_level_type::none);

  KeyGenerator keygen(context);
  SecretKey secret_key = keygen.secret_key();
  PublicKey public_key;
  keygen.create_public_key(public_key);
  RelinKeys relin_keys;
  keygen.create_relin_keys(relin_keys);
  GaloisKeys galois_keys;
  keygen.create_galois_keys(galois_keys);

  Encryptor encryptor(context, public_key);
  CKKSEncoder encoder(context);
  Evaluator evaluator(context, encoder);
  Decryptor decryptor(context, secret_key);

  CKKSEvaluator ckks_evaluator(context, encryptor, decryptor, encoder, evaluator, SCALE, relin_keys, galois_keys);
  GeLUEvaluator gelu_evaluator(ckks_evaluator);

  auto input = read_flat("{(input_dir / 'gelu_input_32768.txt').as_posix()}");
  auto calibration = read_flat("{(cal_dir / 'gelu_calibration_32768.txt').as_posix()}");

  Plaintext plain_input;
  Ciphertext cipher_input;
  Ciphertext cipher_output;
  ckks_evaluator.encoder->encode(input, SCALE, plain_input);
  ckks_evaluator.encryptor->encrypt(plain_input, cipher_input);

  auto start = high_resolution_clock::now();
  gelu_evaluator.gelu(cipher_input, cipher_output);
  auto end = high_resolution_clock::now();

  cout << "[GELU] 32768 takes: " << duration_cast<milliseconds>(end - start).count() << " milliseconds" << endl;
  cout << "Mean Absolute Error: " << ckks_evaluator.calculateMAE(calibration, cipher_output, poly_modulus_degree / 2) << endl;
  return 0;
}}
"""
    if op == "softmax":
        return common_prefix + f"""
int main() {{
  long logN = 16;
  size_t poly_modulus_degree = 1 << logN;
  vector<int> COEFF_MODULI = {{58, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 58}};
  double SCALE = pow(2.0, 40);

  EncryptionParameters parms(scheme_type::ckks);
  parms.set_poly_modulus_degree(poly_modulus_degree);
  parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, COEFF_MODULI));
  SEALContext context(parms, true, sec_level_type::none);

  KeyGenerator keygen(context);
  SecretKey secret_key = keygen.secret_key();
  PublicKey public_key;
  keygen.create_public_key(public_key);
  RelinKeys relin_keys;
  keygen.create_relin_keys(relin_keys);
  GaloisKeys galois_keys;
  keygen.create_galois_keys(galois_keys);

  Encryptor encryptor(context, public_key);
  CKKSEncoder encoder(context);
  Evaluator evaluator(context, encoder);
  Decryptor decryptor(context, secret_key);

  CKKSEvaluator ckks_evaluator(context, encryptor, decryptor, encoder, evaluator, SCALE, relin_keys, galois_keys);
  SoftmaxEvaluator softmax_evaluator(ckks_evaluator);

  auto input = read_flat("{(input_dir / 'softmax_input_128_128.txt').as_posix()}");
  auto calibration = read_flat("{(cal_dir / 'softmax_calibration_128_128.txt').as_posix()}");

  Plaintext plain_input;
  Ciphertext cipher_input;
  Ciphertext cipher_output;
  ckks_evaluator.encoder->encode(input, SCALE, plain_input);
  ckks_evaluator.encryptor->encrypt(plain_input, cipher_input);

  auto start = high_resolution_clock::now();
  softmax_evaluator.softmax(cipher_input, cipher_output, 128);
  auto end = high_resolution_clock::now();

  cout << "[Softmax] 128 x 128 takes: " << duration_cast<milliseconds>(end - start).count() << " milliseconds" << endl;
  cout << "Mean Absolute Error: " << ckks_evaluator.calculateMAE(calibration, cipher_output, 128) << endl;
  return 0;
}}
"""
    if op == "layernorm":
        return common_prefix + f"""
int main() {{
  long logN = 16;
  size_t poly_modulus_degree = 1 << logN;
  vector<int> COEFF_MODULI = {{58, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 58}};
  double SCALE = pow(2.0, 40);

  EncryptionParameters parms(scheme_type::ckks);
  parms.set_poly_modulus_degree(poly_modulus_degree);
  parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, COEFF_MODULI));
  SEALContext context(parms, true, sec_level_type::none);

  KeyGenerator keygen(context);
  SecretKey secret_key = keygen.secret_key();
  PublicKey public_key;
  keygen.create_public_key(public_key);
  RelinKeys relin_keys;
  keygen.create_relin_keys(relin_keys);
  GaloisKeys galois_keys;
  keygen.create_galois_keys(galois_keys);

  Encryptor encryptor(context, public_key);
  CKKSEncoder encoder(context);
  Evaluator evaluator(context, encoder);
  Decryptor decryptor(context, secret_key);

  CKKSEvaluator ckks_evaluator(context, encryptor, decryptor, encoder, evaluator, SCALE, relin_keys, galois_keys);
  LNEvaluator ln_evaluator(ckks_evaluator);

  auto input = read_flat("{(input_dir / 'layernorm_input_16_768.txt').as_posix()}");
  auto calibration = read_flat("{(cal_dir / 'layernorm_calibration_16_768.txt').as_posix()}");

  Plaintext plain_input;
  Ciphertext cipher_input;
  Ciphertext cipher_output;
  ckks_evaluator.encoder->encode(input, SCALE, plain_input);
  ckks_evaluator.encryptor->encrypt(plain_input, cipher_input);

  auto start = high_resolution_clock::now();
  ln_evaluator.layer_norm(cipher_input, cipher_output, 1024);
  auto end = high_resolution_clock::now();

  cout << "[LayerNorm] 16 x 768 takes: " << duration_cast<milliseconds>(end - start).count() << " milliseconds" << endl;
  cout << "Mean Absolute Error: " << ckks_evaluator.calculateMAE(calibration, cipher_output, 768) << endl;
  return 0;
}}
"""
    return f"""
#include <seal/seal.h>
#include <seal/util/uintarith.h>

#include <chrono>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "ckks_evaluator.h"
#include "matrix_mul.h"

using namespace std;
using namespace seal;
using namespace seal::util;
using namespace std::chrono;

int main() {{
  long logN = 13;
  size_t poly_modulus_degree = 1 << logN;
  vector<int> MM_COEFF_MODULI = {{60, 40, 60}};
  double SCALE = pow(2.0, 40);

  EncryptionParameters parms(scheme_type::ckks);
  parms.set_poly_modulus_degree(poly_modulus_degree);
  parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, MM_COEFF_MODULI));
  SEALContext context(parms, true, sec_level_type::none);

  KeyGenerator keygen(context);
  SecretKey secret_key = keygen.secret_key();
  PublicKey public_key;
  keygen.create_public_key(public_key);
  RelinKeys relin_keys;
  keygen.create_relin_keys(relin_keys);
  GaloisKeys galois_keys;

  std::vector<std::uint32_t> rots;
  for (int i = 0; i < logN; i++) {{
    rots.push_back((poly_modulus_degree + exponentiate_uint(2, i)) / exponentiate_uint(2, i));
  }}
  keygen.create_galois_keys(rots, galois_keys);

  Encryptor encryptor(context, public_key);
  CKKSEncoder encoder(context);
  Evaluator evaluator(context, encoder);
  Decryptor decryptor(context, secret_key);

  CKKSEvaluator ckks_evaluator(context, encryptor, decryptor, encoder, evaluator, SCALE, relin_keys, galois_keys);
  MMEvaluator mme(ckks_evaluator);

  auto matrix_4096x768 = mme.readMatrix("{(input_dir / 'matrixmul_input_m_128_n_768_k_64_batch_128.txt').as_posix()}", 4096, 768);
  auto matrix_768x64 = mme.readMatrix("{(input_dir / 'matrix_input_n_768_k_64.txt').as_posix()}", 768, 64);
  auto matrix_4096x768_T = mme.transposeMatrix(matrix_4096x768);
  auto matrix_768x64_T = mme.transposeMatrix(matrix_768x64);

  std::vector<std::vector<double>> row_pack;
  std::vector<double> row_ct(poly_modulus_degree, 0.0);
  for (auto i = 0; i < 64 * 768; i++) {{
    int row = i / 768;
    int col = i % 768;
    row_ct[i % poly_modulus_degree] = matrix_768x64_T[row][col];
    if (i % poly_modulus_degree == (poly_modulus_degree - 1)) {{
      row_pack.push_back(row_ct);
    }}
  }}

  vector<Ciphertext> res;
  auto start = high_resolution_clock::now();
  mme.matrix_mul(matrix_4096x768_T, row_pack, res);
  auto end = high_resolution_clock::now();
  cout << "[MatMul] 4096x768 x 768x64 takes: " << duration_cast<milliseconds>(end - start).count() << " milliseconds" << endl;

  auto matrix_4096x64 = mme.readMatrix("{(cal_dir / 'matrix_output_m_128_k_64_batch_128.txt').as_posix()}", 4096, 64);
  auto matrix_4096x64_T = mme.transposeMatrix(matrix_4096x64);
  double average_err = 0.0;
  Plaintext res_pt;
  vector<double> mm_res;
  ckks_evaluator.decryptor->decrypt(res[0], res_pt);
  ckks_evaluator.encoder->decode(res_pt, mm_res);
  for (auto i = 0; i < 4096; i++) {{
    average_err += fabs(mm_res[i] / 2.0 - matrix_4096x64_T[0][i]);
  }}
  std::cout << "Average error: " << average_err / 4096.0 << std::endl;
  return 0;
}}
"""


def fallback_cmakelists(op: str, nexus_root: Path) -> str:
    src_file = fallback_source_filename(op)
    return f"""
cmake_minimum_required(VERSION 3.10)
project(NEXUSCodexLocal VERSION 1.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${{CMAKE_CURRENT_BINARY_DIR}}/bin)
find_package(SEAL 4.1 REQUIRED)
add_executable(main
  ${{CMAKE_CURRENT_SOURCE_DIR}}/codex_main.cpp
  ${{CMAKE_CURRENT_SOURCE_DIR}}/ckks_evaluator.cpp
  "{(nexus_root / 'src' / src_file).as_posix()}"
)
target_include_directories(main PRIVATE
  "{(nexus_root / 'src').as_posix()}"
  /usr/local/include
)
target_link_libraries(main PRIVATE m pthread SEAL::seal)
"""


def fallback_nexus_run(op: str, meta: dict[str, object], args: argparse.Namespace, nexus_root: Path) -> tuple[Result, str]:
    build_dir = nexus_root / ".codex_local_nexus" / op
    build_dir.mkdir(parents=True, exist_ok=True)
    ckks_src = (nexus_root / "src" / "ckks_evaluator.cpp").read_text()
    ckks_src = ckks_src.replace("compr_mode_type::zstd", "compr_mode_type::none")
    (build_dir / "ckks_evaluator.cpp").write_text(ckks_src)
    (build_dir / "codex_main.cpp").write_text(fallback_main_source(op, nexus_root))
    (build_dir / "CMakeLists.txt").write_text(fallback_cmakelists(op, nexus_root))

    configure_cmd = ["cmake", "."]
    if args.nexus_seal_dir:
        configure_cmd.append(f"-DSEAL_DIR={args.nexus_seal_dir}")
    if args.nexus_cmake_prefix_path:
        configure_cmd.append(f"-DCMAKE_PREFIX_PATH={args.nexus_cmake_prefix_path}")

    run_checked(configure_cmd, cwd=build_dir)
    run_checked(["cmake", "--build", "."], cwd=build_dir)
    output = run_checked([str(build_dir / "bin" / "main")], cwd=build_dir)
    return parse_result(output, str(meta["nexus_label"]), str(meta["metric_label"])), "nexus-source-local"


def try_nexus_run(op: str, meta: dict[str, object], args: argparse.Namespace, nexus_root: Path) -> tuple[Result, str]:
    mode = args.nexus_build_mode
    if mode in ("official", "auto"):
        try:
            return official_nexus_run(op, meta, args, nexus_root)
        except RuntimeError as exc:
            if mode == "official":
                raise
            print(f"Official NEXUS build for {op} failed, falling back: {exc}", file=sys.stderr)

    if not bool(meta["fallback_supported"]):
        raise RuntimeError(f"No local fallback build is supported for op '{op}' on this machine")
    return fallback_nexus_run(op, meta, args, nexus_root)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run official NEXUS CPU benchmarks and local baseline benchmarks in one command."
    )
    parser.add_argument("--nexus-root", required=True, help="Path to official NEXUS checkout")
    parser.add_argument(
        "--mini-ppti-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Path to the mini-ppti repo root",
    )
    parser.add_argument(
        "--ops",
        default="all",
        help="Comma-separated subset of: matmul,argmax,gelu,layernorm,softmax or all",
    )
    parser.add_argument("--impl", default="baseline", choices=["baseline", "optimized"])
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--output-csv", help="Optional CSV path for the joined comparison table")
    parser.add_argument("--skipped-csv", help="Optional CSV path for skipped ops and failure reasons")
    parser.add_argument("--nexus-seal-dir", help="Optional SEAL_DIR passed to NEXUS CMake configure")
    parser.add_argument(
        "--nexus-cmake-prefix-path",
        help="Optional CMAKE_PREFIX_PATH passed to NEXUS CMake configure",
    )
    parser.add_argument(
        "--nexus-build-mode",
        default="auto",
        choices=["auto", "official", "nexus-source-local"],
        help="Use the official NEXUS build, or fall back to a minimal local build from NEXUS source files.",
    )
    args = parser.parse_args()

    mini_root = Path(args.mini_ppti_root).resolve()
    nexus_root = Path(args.nexus_root).resolve()
    baseline_binary = mini_root / "cpp" / "build" / "mini_ppti"

    if not (nexus_root / "src" / "main.cpp").exists():
        raise SystemExit(f"NEXUS main.cpp not found under {nexus_root}")
    if not baseline_binary.exists():
        raise SystemExit(f"Baseline binary not found: {baseline_binary}")

    selected_ops = list(OPS) if args.ops == "all" else [op.strip() for op in args.ops.split(",") if op.strip()]
    unknown_ops = sorted(set(selected_ops) - set(OPS))
    if unknown_ops:
        raise SystemExit(f"Unsupported --ops entries: {', '.join(unknown_ops)}")

    rows = []
    failures = []
    for op in selected_ops:
        meta = OPS[op]
        print(f"Running {op}...", flush=True)
        try:
            if args.nexus_build_mode == "nexus-source-local":
                nexus_result, nexus_mode = fallback_nexus_run(op, meta, args, nexus_root)
            else:
                nexus_result, nexus_mode = try_nexus_run(op, meta, args, nexus_root)
        except RuntimeError as exc:
            failures.append((op, str(exc)))
            continue

        baseline_cmd = [
            str(baseline_binary),
            "--op",
            str(meta["baseline_op"]),
            "--impl",
            args.impl,
            "--nexus-input-dir",
            str(nexus_root / "data" / "input"),
            "--nexus-calibration-dir",
            str(nexus_root / "data" / "calibration"),
            "--trials",
            str(args.trials),
            "--warmup",
            str(args.warmup),
        ]
        try:
            baseline_output = run_checked(baseline_cmd, cwd=mini_root / "cpp")
            baseline_result = parse_result(
                baseline_output,
                str(meta["baseline_label"]),
                str(meta["metric_label"]),
            )
        except RuntimeError as exc:
            failures.append((op, str(exc)))
            continue

        rows.append(
            {
                "op": op,
                "nexus_ms": nexus_result.runtime_ms,
                "baseline_ms": baseline_result.runtime_ms,
                "baseline_vs_nexus": baseline_result.runtime_ms / nexus_result.runtime_ms,
                "nexus_metric_name": nexus_result.metric_name,
                "nexus_metric_value": nexus_result.metric_value,
                "baseline_metric_name": baseline_result.metric_name,
                "baseline_metric_value": baseline_result.metric_value,
                "nexus_mode": nexus_mode,
            }
        )

    if rows:
        print()
        print_table(rows)
        if args.output_csv:
            output_csv = Path(args.output_csv)
            if not output_csv.is_absolute():
                output_csv = mini_root / output_csv
            output_csv.parent.mkdir(parents=True, exist_ok=True)
            write_csv(output_csv, rows)
            print(f"\nWrote comparison CSV to {output_csv}")

    if failures:
        print("\nSkipped ops:", file=sys.stderr)
        for op, reason in failures:
            print(f"- {op}: {reason.splitlines()[0]}", file=sys.stderr)
        if args.skipped_csv:
            skipped_csv = Path(args.skipped_csv)
            if not skipped_csv.is_absolute():
                skipped_csv = mini_root / skipped_csv
            skipped_csv.parent.mkdir(parents=True, exist_ok=True)
            write_failures_csv(skipped_csv, failures)
            print(f"Wrote skipped-op CSV to {skipped_csv}", file=sys.stderr)

    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
