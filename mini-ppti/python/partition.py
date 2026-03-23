from graph import Graph
from ops import Domain, default_domain_for_op


def assign_domains(graph: Graph) -> None:
    """
    Assign a default execution domain to each node.

    Rules for now:
    - input/output -> plain
    - linear ops   -> HE
    - nonlinear ops -> MPC

    This is a baseline rule-based partitioner.
    Later we can replace it with smarter scheduling logic.
    """
    for node in graph.topological_order():
        node.domain = default_domain_for_op(node.kind)


def count_domains(graph: Graph) -> dict[str, int]:
    counts = {
        Domain.HE.value: 0,
        Domain.MPC.value: 0,
        Domain.PLAIN.value: 0,
        "unassigned": 0,
    }

    for node in graph.topological_order():
        if node.domain is None:
            counts["unassigned"] += 1
        else:
            counts[node.domain.value] += 1

    return counts