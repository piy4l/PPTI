from typing import Any

from graph import Graph
from ops import Domain, OpKind


def estimate_cost(graph: Graph) -> dict[str, Any]:
    """
    Slightly improved symbolic cost model.

    Counts:
    - domain-boundary conversions
    - rough truncation count
    - rough rotation count
    - rough ct-ct multiply count
    """
    stats = {
        "num_nodes": 0,
        "num_he_nodes": 0,
        "num_mpc_nodes": 0,
        "num_plain_nodes": 0,
        "num_conversions": 0,
        "num_truncations": 0,
        "num_rotations": 0,
        "num_ct_ct_mults": 0,
        "boundary_edges": [],
    }

    for node in graph.topological_order():
        stats["num_nodes"] += 1

        if node.domain == Domain.HE:
            stats["num_he_nodes"] += 1
        elif node.domain == Domain.MPC:
            stats["num_mpc_nodes"] += 1
        elif node.domain == Domain.PLAIN:
            stats["num_plain_nodes"] += 1

        if node.domain == Domain.HE:
            if node.kind == OpKind.MATMUL:
                stats["num_rotations"] += 2

            elif node.kind == OpKind.SUM:
                stats["num_rotations"] += 1

            elif node.kind == OpKind.MUL:
                stats["num_ct_ct_mults"] += 1
                stats["num_truncations"] += 1

        for pred in graph.predecessors(node.name):
            if pred.domain is None or node.domain is None:
                continue

            if pred.domain != node.domain:
                stats["num_conversions"] += 1
                stats["boundary_edges"].append(
                    {
                        "from": pred.name,
                        "to": node.name,
                        "from_domain": pred.domain.value,
                        "to_domain": node.domain.value,
                    }
                )

    return stats


def format_cost_report(stats: dict[str, Any]) -> str:
    lines = [
        "=== Cost Report ===",
        f"Nodes            : {stats['num_nodes']}",
        f"HE nodes         : {stats['num_he_nodes']}",
        f"MPC nodes        : {stats['num_mpc_nodes']}",
        f"Plain nodes      : {stats['num_plain_nodes']}",
        f"Conversions      : {stats['num_conversions']}",
        f"Truncations      : {stats['num_truncations']}",
        f"Rotations        : {stats['num_rotations']}",
        f"CT-CT multiplies : {stats['num_ct_ct_mults']}",
    ]

    if stats["boundary_edges"]:
        lines.append("Boundary edges:")
        for edge in stats["boundary_edges"]:
            lines.append(
                f"  - {edge['from']} ({edge['from_domain']}) "
                f"-> {edge['to']} ({edge['to_domain']})"
            )
    else:
        lines.append("Boundary edges: none")

    return "\n".join(lines)