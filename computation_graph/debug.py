from typing import Callable, Text, Tuple

import gamla

from computation_graph import base_types, graph, run, visualization

_ComputationTrace = Callable[
    [base_types.ComputationNode],
    Tuple[Tuple[base_types.ComputationNode, base_types.ComputationResult]],
]


def _node_computation_trace(node_to_results: run.NodeToResults) -> _ComputationTrace:
    return gamla.compose_many_to_one(
        [
            gamla.pair_right(gamla.compose_left(node_to_results, gamla.head)),
            gamla.compose_left(node_to_results, dict.values, gamla.head, dict.items),
        ],
        gamla.prefix,
    )


@gamla.curry
def debugger(
    filename: Text, edges: base_types.GraphType, node_to_results: run.NodeToResults
):
    return visualization.serialize_computation_trace(filename)(
        [edges, _node_computation_trace(node_to_results)(graph.infer_graph_sink(edges))]
    )
