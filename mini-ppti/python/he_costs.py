from he_bridge import HEBenchmarkRunner


class MeasuredHECostModel:
    def __init__(self, runner: HEBenchmarkRunner):
        self.runner = runner
        self.cache: dict[tuple, float] = {}

    def _cached(self, key: tuple, fetch_fn) -> float:
        if key not in self.cache:
            row = fetch_fn()
            self.cache[key] = float(row["avg_ms"])
        return self.cache[key]

    def encrypt_only_cost(self, n: int, trials: int = 10, warmup: int = 3) -> float:
        key = ("encrypt_only", n, trials, warmup)
        return self._cached(
            key,
            lambda: self.runner.benchmark_encrypt_only(n=n, trials=trials, warmup=warmup),
        )

    def decrypt_only_cost(self, n: int, trials: int = 10, warmup: int = 3) -> float:
        key = ("decrypt_only", n, trials, warmup)
        return self._cached(
            key,
            lambda: self.runner.benchmark_decrypt_only(n=n, trials=trials, warmup=warmup),
        )

    def encrypt_decrypt_cost(self, n: int, trials: int = 10, warmup: int = 3) -> float:
        key = ("encrypt_decrypt", n, trials, warmup)
        return self._cached(
            key,
            lambda: self.runner.benchmark_encrypt_decrypt(n=n, trials=trials, warmup=warmup),
        )

    def rotate_cost(self, n: int, steps: int = 1, trials: int = 10, warmup: int = 3) -> float:
        key = ("rotate", n, steps, trials, warmup)
        return self._cached(
            key,
            lambda: self.runner.benchmark_rotate(n=n, steps=steps, trials=trials, warmup=warmup),
        )

    def mul_plain_cost(self, n: int, scalar: float = 2.0, trials: int = 10, warmup: int = 3) -> float:
        key = ("mul_plain", n, scalar, trials, warmup)
        return self._cached(
            key,
            lambda: self.runner.benchmark_mul_plain(n=n, scalar=scalar, trials=trials, warmup=warmup),
        )

    def add_plain_cost(self, n: int, add_const: float = 5.0, trials: int = 10, warmup: int = 3) -> float:
        key = ("add_plain", n, add_const, trials, warmup)
        return self._cached(
            key,
            lambda: self.runner.benchmark_add_plain(n=n, add_const=add_const, trials=trials, warmup=warmup),
        )
    
    def mul_ct_ct_cost(self, n: int, trials: int = 10, warmup: int = 3) -> float:
        key = ("mul_ct_ct", n, trials, warmup)
        return self._cached(
            key,
            lambda: self.runner.benchmark_mul_ct_ct(n=n, trials=trials, warmup=warmup),
        )

    def toy_transformer_block_cost(self, n: int, trials: int = 10, warmup: int = 3) -> float:
        key = ("toy_transformer_block", n, trials, warmup)
        return self._cached(
            key,
            lambda: self.runner.benchmark_toy_transformer_block(
                n=n, trials=trials, warmup=warmup
            ),
        )
