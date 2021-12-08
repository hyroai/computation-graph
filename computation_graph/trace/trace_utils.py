from typing import Callable, FrozenSet, Tuple

import gamla

from computation_graph import base_types


def node_computation_trace(
    node_to_results: Callable, node: base_types.ComputationNode
) -> FrozenSet[Tuple[base_types.ComputationNode, base_types.ComputationResult]]:
    results = node_to_results(node)
    try:
        first_result = gamla.head(results)
    except StopIteration:
        return frozenset()
    return frozenset(
        [
            (node, first_result),
            *gamla.pipe(results, dict.values, gamla.head, dict.items),
        ]
    )


def is_edge_participating(in_trace_nodes):
    def is_edge_participating(edge):
        return in_trace_nodes(edge.destination) and gamla.anymap(in_trace_nodes)(
            [edge.source, *edge.args]
        )

    return is_edge_participating


def get_edge_label(edge: base_types.ComputationEdge) -> str:
    if edge.key in (None, "args"):
        return ""
    if edge.key == "first_input":
        return str(edge.priority)
    return edge.key or ""
