from typing import Iterable, Tuple

import gamla

from computation_graph import base_types, graph, run, visualization


def _node_computation_trace(
    node_to_results: run.NodeToResults, node: base_types.ComputationNode
) -> Iterable[Tuple[base_types.ComputationNode, base_types.ComputationResult]]:
    return frozenset(
        [
            (node, gamla.head(node_to_results(node))),
            *gamla.pipe(node, node_to_results, dict.values, gamla.head, dict.items),
        ]
    )


@gamla.curry
def serialize_computation_trace(
    filename: str,
    graph_instance: base_types.GraphType,
    node_to_results: run.NodeToResults,
):
    gviz = visualization.union_graphviz(
        [
            visualization.computation_graph_to_graphviz(graph_instance),
            visualization.computation_trace_to_graphviz(
                _node_computation_trace(
                    node_to_results, graph.infer_graph_sink(graph_instance)
                )
            ),
        ]
    )
    gviz.layout(prog="dot")
    gviz.write(filename)
