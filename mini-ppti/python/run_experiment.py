import argparse

from fusion import estimate_fused_cost, find_fusion_groups, format_fusion_report
from graph import Graph, Node
from ops import OpKind
from partition import assign_domains, count_domains
from cost_model import estimate_cost, format_cost_report
from he_bridge import HEBenchmarkRunner
from he_costs import MeasuredHECostModel

def build_toy_graph() -> Graph:
    """
    A tiny LayerNorm -> FC -> GeLU-style toy graph.

    This is not numerically exact Transformer code.
    It is a structural graph for partitioning/fusion experiments.
    """
    nodes = [
        Node("input", OpKind.INPUT),

        # LayerNorm-like region
        Node("ln_sum", OpKind.SUM, inputs=["input"]),
        Node("ln_var_mul", OpKind.MUL, inputs=["ln_sum"]),
        Node("ln_rsqrt", OpKind.RSQRT, inputs=["ln_var_mul"]),
        Node("ln_mul_gamma", OpKind.MUL, inputs=["ln_rsqrt"]),
        Node("ln_add_beta", OpKind.ADD, inputs=["ln_mul_gamma"]),

        # Fully connected layer
        Node("fc", OpKind.MATMUL, inputs=["ln_add_beta"]),

        # GeLU-like polynomial region
        Node("gelu_mul1", OpKind.MUL, inputs=["fc"]),
        Node("gelu_mul2", OpKind.MUL, inputs=["gelu_mul1"]),
        Node("gelu_add", OpKind.ADD, inputs=["gelu_mul2"]),

        Node("output", OpKind.OUTPUT, inputs=["gelu_add"]),
    ]
    return Graph(nodes)


def build_transformer_block_graph() -> Graph:
    """
    A small Transformer-block style graph.

    This is still structural rather than numerically exact. The goal is to
    expose the main regions that matter for partitioning:
    - Q/K/V projections
    - attention score/value flow
    - residual and layernorm boundaries
    - MLP expansion and projection
    """
    nodes = [
        Node("input", OpKind.INPUT),

        # Self-attention projections
        Node("q_proj", OpKind.MATMUL, inputs=["input"]),
        Node("k_proj", OpKind.MATMUL, inputs=["input"]),
        Node("v_proj", OpKind.MATMUL, inputs=["input"]),

        # Attention score path
        Node("attn_scores", OpKind.MATMUL, inputs=["q_proj", "k_proj"]),
        Node("attn_scale", OpKind.MUL, inputs=["attn_scores"]),
        Node("attn_softmax", OpKind.EXP, inputs=["attn_scale"]),
        Node("attn_weighted_values", OpKind.MATMUL, inputs=["attn_softmax", "v_proj"]),
        Node("attn_out_proj", OpKind.MATMUL, inputs=["attn_weighted_values"]),
        Node("attn_residual", OpKind.ADD, inputs=["attn_out_proj", "input"]),

        # LayerNorm-like boundary
        Node("ln1_sum", OpKind.SUM, inputs=["attn_residual"]),
        Node("ln1_var_mul", OpKind.MUL, inputs=["ln1_sum"]),
        Node("ln1_rsqrt", OpKind.RSQRT, inputs=["ln1_var_mul"]),
        Node("ln1_mul_gamma", OpKind.MUL, inputs=["ln1_rsqrt"]),
        Node("ln1_add_beta", OpKind.ADD, inputs=["ln1_mul_gamma"]),

        # MLP block
        Node("mlp_fc1", OpKind.MATMUL, inputs=["ln1_add_beta"]),
        Node("mlp_gelu_mul1", OpKind.MUL, inputs=["mlp_fc1"]),
        Node("mlp_gelu_mul2", OpKind.MUL, inputs=["mlp_gelu_mul1"]),
        Node("mlp_gelu_add", OpKind.ADD, inputs=["mlp_gelu_mul2"]),
        Node("mlp_fc2", OpKind.MATMUL, inputs=["mlp_gelu_add"]),
        Node("mlp_residual", OpKind.ADD, inputs=["mlp_fc2", "attn_residual"]),

        # Final LayerNorm-like boundary
        Node("ln2_sum", OpKind.SUM, inputs=["mlp_residual"]),
        Node("ln2_var_mul", OpKind.MUL, inputs=["ln2_sum"]),
        Node("ln2_rsqrt", OpKind.RSQRT, inputs=["ln2_var_mul"]),
        Node("ln2_mul_gamma", OpKind.MUL, inputs=["ln2_rsqrt"]),
        Node("ln2_add_beta", OpKind.ADD, inputs=["ln2_mul_gamma"]),

        Node("output", OpKind.OUTPUT, inputs=["ln2_add_beta"]),
    ]
    return Graph(nodes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run partitioning and HE cost experiments on toy graphs."
    )
    parser.add_argument(
        "--graph",
        choices=["toy", "transformer_block"],
        default="toy",
        help="Which graph to analyze.",
    )
    parser.add_argument(
        "--fusion",
        action="store_true",
        help="Also print symbolic HE fusion analysis.",
    )
    return parser.parse_args()


def build_graph(name: str) -> Graph:
    if name == "toy":
        return build_toy_graph()
    if name == "transformer_block":
        return build_transformer_block_graph()
    raise ValueError(f"Unsupported graph: {name}")


def main() -> None:
    args = parse_args()
    graph = build_graph(args.graph)

    print(f"=== Initial Graph ({args.graph}) ===")
    print(graph.summary())
    print()

    assign_domains(graph)

    runner = HEBenchmarkRunner("../cpp/build/mini_ppti")
    he_cost_model = MeasuredHECostModel(runner)

    print("=== Domain-Assigned Graph ===")
    print(graph.summary())
    print()

    domain_counts = count_domains(graph)
    print("=== Domain Counts ===")
    for key, value in domain_counts.items():
        print(f"{key:10s}: {value}")
    print()

    stats = estimate_cost(graph, he_cost_model=he_cost_model)
    print(format_cost_report(stats))

    if args.fusion:
        print()
        groups = find_fusion_groups(graph)
        fused_stats = estimate_fused_cost(graph, groups)
        print(format_fusion_report(groups, fused_stats))


if __name__ == "__main__":
    main()
