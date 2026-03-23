from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ops import Domain, OpKind


@dataclass
class Node:
    name: str
    kind: OpKind
    inputs: List[str] = field(default_factory=list)
    domain: Optional[Domain] = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Node name cannot be empty")


class Graph:
    def __init__(self, nodes: List[Node]) -> None:
        self.nodes = nodes
        self.node_map: Dict[str, Node] = {}

        for node in nodes:
            if node.name in self.node_map:
                raise ValueError(f"Duplicate node name detected: {node.name}")
            self.node_map[node.name] = node

        self._validate_inputs()

    def _validate_inputs(self) -> None:
        for node in self.nodes:
            for input_name in node.inputs:
                if input_name not in self.node_map:
                    raise ValueError(
                        f"Node '{node.name}' references missing input '{input_name}'"
                    )

    def get_node(self, name: str) -> Node:
        if name not in self.node_map:
            raise KeyError(f"Node not found: {name}")
        return self.node_map[name]

    def predecessors(self, node_name: str) -> List[Node]:
        node = self.get_node(node_name)
        return [self.get_node(input_name) for input_name in node.inputs]

    def successors(self, node_name: str) -> List[Node]:
        result: List[Node] = []
        for node in self.nodes:
            if node_name in node.inputs:
                result.append(node)
        return result

    def topological_order(self) -> List[Node]:
        return self.nodes

    def summary(self) -> str:
        lines = []
        for node in self.nodes:
            domain = node.domain.value if node.domain else "unassigned"
            inputs = ", ".join(node.inputs) if node.inputs else "-"
            lines.append(
                f"{node.name:20s} kind={node.kind.value:8s} "
                f"domain={domain:10s} inputs=[{inputs}]"
            )
        return "\n".join(lines)