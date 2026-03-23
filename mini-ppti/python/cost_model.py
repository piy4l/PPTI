from typing import Any

from graph import Graph
from ops import Domain, OpKind


def estimate_cost(graph: Graph) -> dict[str, Any]:
    """
    Simple symbolic cost model.

    For now we estimate:
    - number of HE<->MPC / HE<->PLAIN / MPC<->PLAIN boundaries
    - rough truncation count
    - rough rotation count
    - ciphertext-ciphertext multiplication count

    This is intentionally simple. Later we will replace parts of it
    with measured costs from the C++ OpenFHE backend.
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

        # Very rough op-level costs inside HE
        if node.domain == Domain.HE:
            if node.kind == OpKind.MATMUL:
                # MatMul is usually the heaviest HE op family.
                # Start with a symbolic rotation estimate of 1.
                # Later we will replace this with dimension-aware logic.
                stats["num_rotations"] += 1

            if node.kind == OpKind.MUL:
                # Treat HE multiply as a ct-ct multiply candidate.
                # Later we can distinguish ct-pt vs ct-ct.
                stats["num_ct_ct_mults"] += 1
                stats["num_truncations"] += 1

        # Count domain-boundary conversions on edges
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