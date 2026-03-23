from graph import Graph, Node
from ops import OpKind
from partition import assign_domains, count_domains
from cost_model import estimate_cost, format_cost_report


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
    print("=== Domain Counts ===")
    for key, value in domain_counts.items():
        print(f"{key:10s}: {value}")
    print()

    stats = estimate_cost(graph)
    print(format_cost_report(stats))


if __name__ == "__main__":
    main()