from dataclasses import dataclass
from typing import List

from graph import Graph, Node
from ops import Domain, OpKind


FUSIBLE_HE_OPS = {
    OpKind.MATMUL,
    OpKind.ADD,
    OpKind.MUL,
    OpKind.SUM,
}


@dataclass
class FusionGroup:
    name: str
    node_names: List[str]
    domain: Domain

    def __str__(self) -> str:
        return f"{self.name}: domain={self.domain.value}, nodes={self.node_names}"


def is_he_fusible(node: Node) -> bool:
    return node.domain == Domain.HE and node.kind in FUSIBLE_HE_OPS


def can_fuse_pair(prev_node: Node, curr_node: Node, graph: Graph) -> bool:
    """
    Conservative fusion rule for now.

    We only fuse prev -> curr if:
    1. both are HE-fusible
    2. curr has exactly one input
    3. curr directly depends on prev
    4. prev has exactly one successor

    This keeps fusion simple and safe for a linear-chain toy graph.
    """
    if not is_he_fusible(prev_node) or not is_he_fusible(curr_node):
        return False

    if len(curr_node.inputs) != 1:
        return False

    if curr_node.inputs[0] != prev_node.name:
        return False

    succs = graph.successors(prev_node.name)
    if len(succs) != 1 or succs[0].name != curr_node.name:
        return False

    return True


def find_fusion_groups(graph: Graph) -> List[FusionGroup]:
    nodes = graph.topological_order()
    groups: List[FusionGroup] = []

    current_group: List[Node] = []

    def flush_group() -> None:
        nonlocal current_group
        if not current_group:
            return

        if len(current_group) == 1:
            node = current_group[0]
            groups.append(
                FusionGroup(
                    name=f"group_{node.name}",
                    node_names=[node.name],
                    domain=node.domain,
                )
            )
        else:
            names = [node.name for node in current_group]
            groups.append(
                FusionGroup(
                    name=f"fused_{names[0]}_to_{names[-1]}",
                    node_names=names,
                    domain=current_group[0].domain,
                )
            )
        current_group = []

    for node in nodes:
        if not current_group:
            current_group = [node]
            continue

        prev = current_group[-1]
        if can_fuse_pair(prev, node, graph):
            current_group.append(node)
        else:
            flush_group()
            current_group = [node]

    flush_group()
    return groups


def estimate_fused_cost(graph: Graph, groups: List[FusionGroup]) -> dict:
    """
    Very simple group-level cost model.

    Idea:
    - conversion cost counts boundaries between adjacent groups
    - truncation/rotation/ct-ct multiply cost inside HE groups is reduced
      compared to the unfused node-wise estimate

    This is symbolic and intentionally conservative.
    """
    group_map = {}
    for idx, group in enumerate(groups):
        for node_name in group.node_names:
            group_map[node_name] = idx

    stats = {
        "num_groups": len(groups),
        "num_group_conversions": 0,
        "num_group_truncations": 0,
        "num_group_rotations": 0,
        "num_group_ct_ct_mults": 0,
        "group_boundary_edges": [],
    }

    # Count boundaries between groups
    for node in graph.topological_order():
        for pred in graph.predecessors(node.name):
            g1 = group_map[pred.name]
            g2 = group_map[node.name]

            if g1 == g2:
                continue

            pred_group = groups[g1]
            node_group = groups[g2]

            if pred_group.domain != node_group.domain:
                stats["num_group_conversions"] += 1
                stats["group_boundary_edges"].append(
                    {
                        "from_group": pred_group.name,
                        "to_group": node_group.name,
                        "from_domain": pred_group.domain.value,
                        "to_domain": node_group.domain.value,
                    }
                )

    # Symbolic in-group HE cost
    for group in groups:
        if group.domain != Domain.HE:
            continue

        he_nodes = [graph.get_node(name) for name in group.node_names]

        has_matmul = any(node.kind == OpKind.MATMUL for node in he_nodes)
        mul_count = sum(1 for node in he_nodes if node.kind == OpKind.MUL)

        if has_matmul:
            stats["num_group_rotations"] += 1

        if mul_count > 0:
            stats["num_group_ct_ct_mults"] += mul_count
            # key symbolic benefit: one fused HE region pays fewer truncation points
            stats["num_group_truncations"] += 1

    return stats


def format_fusion_report(groups: List[FusionGroup], fused_stats: dict) -> str:
    lines = ["=== Fusion Report ===", "Fusion groups:"]

    for group in groups:
        lines.append(f"  - {group}")

    lines.extend(
        [
            "",
            "=== Fused Cost Report ===",
            f"Groups                 : {fused_stats['num_groups']}",
            f"Group conversions      : {fused_stats['num_group_conversions']}",
            f"Group truncations      : {fused_stats['num_group_truncations']}",
            f"Group rotations        : {fused_stats['num_group_rotations']}",
            f"Group ct-ct multiplies : {fused_stats['num_group_ct_ct_mults']}",
        ]
    )

    if fused_stats["group_boundary_edges"]:
        lines.append("Group boundary edges:")
        for edge in fused_stats["group_boundary_edges"]:
            lines.append(
                f"  - {edge['from_group']} ({edge['from_domain']}) "
                f"-> {edge['to_group']} ({edge['to_domain']})"
            )
    else:
        lines.append("Group boundary edges: none")

    return "\n".join(lines)