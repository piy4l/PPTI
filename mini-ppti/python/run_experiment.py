from graph import Graph, Node
from ops import OpKind
from partition import assign_domains, count_domains
from cost_model import estimate_cost, format_cost_report
from fusion import find_fusion_groups, estimate_fused_cost, format_fusion_report


def build_toy_graph() -> Graph:
    """
    A more realistic toy graph for:
    LayerNorm -> FC -> GeLU

    This is still symbolic, but it separates the main operator regions better.
    """
    nodes = [
        Node("input", OpKind.INPUT),

        # LayerNorm-ish decomposition
        Node("ln_mean_sum", OpKind.SUM, inputs=["input"]),
        Node("ln_center_add", OpKind.ADD, inputs=["ln_mean_sum"]),
        Node("ln_var_mul", OpKind.MUL, inputs=["ln_center_add"]),
        Node("ln_var_sum", OpKind.SUM, inputs=["ln_var_mul"]),
        Node("ln_rsqrt", OpKind.RSQRT, inputs=["ln_var_sum"]),
        Node("ln_scale_mul", OpKind.MUL, inputs=["ln_rsqrt"]),
        Node("ln_shift_add", OpKind.ADD, inputs=["ln_scale_mul"]),

        # Fully connected
        Node("fc", OpKind.MATMUL, inputs=["ln_shift_add"]),
        Node("fc_bias_add", OpKind.ADD, inputs=["fc"]),

        # GeLU-ish polynomial region
        Node("gelu_mul1", OpKind.MUL, inputs=["fc_bias_add"]),
        Node("gelu_add1", OpKind.ADD, inputs=["gelu_mul1"]),
        Node("gelu_mul2", OpKind.MUL, inputs=["gelu_add1"]),
        Node("gelu_add2", OpKind.ADD, inputs=["gelu_mul2"]),

        Node("output", OpKind.OUTPUT, inputs=["gelu_add2"]),
    ]
    return Graph(nodes)


def print_domain_counts(domain_counts: dict[str, int]) -> None:
    print("=== Domain Counts ===")
    for key, value in domain_counts.items():
        print(f"{key:10s}: {value}")
    print()


def print_comparison(unfused: dict, fused: dict) -> None:
    print("=== Before vs After Fusion ===")
    print(
        f"Conversions      : {unfused['num_conversions']} -> {fused['num_group_conversions']}"
    )
    print(
        f"Truncations      : {unfused['num_truncations']} -> {fused['num_group_truncations']}"
    )
    print(
        f"Rotations        : {unfused['num_rotations']} -> {fused['num_group_rotations']}"
    )
    print(
        f"CT-CT multiplies : {unfused['num_ct_ct_mults']} -> {fused['num_group_ct_ct_mults']}"
    )
    print()


def main() -> None:
    graph = build_toy_graph()

    print("=== Initial Graph ===")
    print(graph.summary())
    print()

    assign_domains(graph)

    print("=== Domain-Assigned Graph ===")
    print(graph.summary())
    print()

    domain_counts = count_domains(graph)
    print_domain_counts(domain_counts)

    stats = estimate_cost(graph)
    print(format_cost_report(stats))
    print()

    groups = find_fusion_groups(graph)
    fused_stats = estimate_fused_cost(graph, groups)
    print(format_fusion_report(groups, fused_stats))
    print()

    print_comparison(stats, fused_stats)


if __name__ == "__main__":
    main()