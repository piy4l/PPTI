import csv
import io
import subprocess
from pathlib import Path
from typing import Any


class HEBenchmarkError(RuntimeError):
    pass


class HEBenchmarkRunner:
    def __init__(self, binary_path: str):
        self.binary_path = str(Path(binary_path).expanduser())

    def _run(self, args: list[str]) -> list[dict[str, Any]]:
        cmd = [self.binary_path] + args

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            raise HEBenchmarkError(
                f"Benchmark command failed.\n"
                f"Command: {' '.join(cmd)}\n"
                f"STDOUT:\n{e.stdout}\n"
                f"STDERR:\n{e.stderr}"
            ) from e
        except FileNotFoundError as e:
            raise HEBenchmarkError(f"Binary not found: {self.binary_path}") from e

        output = result.stdout.strip()
        if not output:
            raise HEBenchmarkError("Benchmark returned empty output")

        reader = csv.DictReader(io.StringIO(output))
        rows = list(reader)

        if not rows:
            raise HEBenchmarkError(
                f"Could not parse CSV output from benchmark.\nOutput was:\n{output}"
            )

        return rows

    def benchmark_encrypt_only(self, n: int, trials: int = 10, warmup: int = 3) -> dict[str, Any]:
        rows = self._run([
            "--op", "encrypt_only",
            "--n", str(n),
            "--trials", str(trials),
            "--warmup", str(warmup),
            "--csv-only",
        ])
        return rows[0]

    def benchmark_decrypt_only(self, n: int, trials: int = 10, warmup: int = 3) -> dict[str, Any]:
        rows = self._run([
            "--op", "decrypt_only",
            "--n", str(n),
            "--trials", str(trials),
            "--warmup", str(warmup),
            "--csv-only",
        ])
        return rows[0]

    def benchmark_encrypt_decrypt(self, n: int, trials: int = 10, warmup: int = 3) -> dict[str, Any]:
        rows = self._run([
            "--op", "encrypt_decrypt",
            "--n", str(n),
            "--trials", str(trials),
            "--warmup", str(warmup),
            "--csv-only",
        ])
        return rows[0]

    def benchmark_rotate(self, n: int, steps: int = 1, trials: int = 10, warmup: int = 3) -> dict[str, Any]:
        rows = self._run([
            "--op", "rotate",
            "--n", str(n),
            "--steps", str(steps),
            "--trials", str(trials),
            "--warmup", str(warmup),
            "--csv-only",
        ])
        return rows[0]

    def benchmark_mul_plain(self, n: int, scalar: float = 2.0, trials: int = 10, warmup: int = 3) -> dict[str, Any]:
        rows = self._run([
            "--op", "mul_plain",
            "--n", str(n),
            "--scalar", str(scalar),
            "--trials", str(trials),
            "--warmup", str(warmup),
            "--csv-only",
        ])
        return rows[0]

    def benchmark_add_plain(self, n: int, add_const: float = 5.0, trials: int = 10, warmup: int = 3) -> dict[str, Any]:
        rows = self._run([
            "--op", "add_plain",
            "--n", str(n),
            "--add-const", str(add_const),
            "--trials", str(trials),
            "--warmup", str(warmup),
            "--csv-only",
        ])
        return rows[0]
    
    def benchmark_mul_ct_ct(self, n: int, trials: int = 10, warmup: int = 3) -> dict[str, Any]:
        rows = self._run([
            "--op", "mul_ct_ct",
            "--n", str(n),
            "--trials", str(trials),
            "--warmup", str(warmup),
            "--csv-only",
        ])
        return rows[0]

    def benchmark_toy_transformer_block(self, n: int, trials: int = 10, warmup: int = 3) -> dict[str, Any]:
        rows = self._run([
            "--op", "toy_transformer_block",
            "--n", str(n),
            "--trials", str(trials),
            "--warmup", str(warmup),
            "--csv-only",
        ])
        return rows[0]
