from he_bridge import HEBenchmarkRunner
from he_costs import MeasuredHECostModel


def main() -> None:
    runner = HEBenchmarkRunner("../cpp/build/mini_ppti")
    costs = MeasuredHECostModel(runner)

    n = 16
    trials = 10
    warmup = 3

    print("Measured HE costs")
    print(f"encrypt_only({n})    = {costs.encrypt_only_cost(n, trials, warmup):.6f} ms")
    print(f"decrypt_only({n})    = {costs.decrypt_only_cost(n, trials, warmup):.6f} ms")
    print(f"encrypt_decrypt({n}) = {costs.encrypt_decrypt_cost(n, trials, warmup):.6f} ms")
    print(f"add_plain({n})       = {costs.add_plain_cost(n, 5.0, trials, warmup):.6f} ms")
    print(f"mul_plain({n})       = {costs.mul_plain_cost(n, 2.0, trials, warmup):.6f} ms")
    print(f"mul_ct_ct({n})       = {costs.mul_ct_ct_cost(n, trials, warmup):.6f} ms")
    print(f"rotate({n}, 1)       = {costs.rotate_cost(n, 1, trials, warmup):.6f} ms")
    print(f"toy_transformer_block({n}) = {costs.toy_transformer_block_cost(n, trials, warmup):.6f} ms")


if __name__ == "__main__":
    main()
