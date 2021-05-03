import logging
from typing import FrozenSet, Tuple

import gamla

from computation_graph import base_types, graph, run, visualization


def _node_computation_trace(
    node_to_results: run.NodeToResults, node: base_types.ComputationNode
) -> FrozenSet[Tuple[base_types.ComputationNode, base_types.ComputationResult]]:
    return frozenset(
        [
            (node, gamla.head(node_to_results(node))),
            *gamla.pipe(node, node_to_results, dict.values, gamla.head, dict.items),
        ]
    )


@gamla.curry
def gviz_computation_trace(
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


def _clean_for_mermaid_name(obj):
    return str(obj).replace('"', "")


@gamla.star
def _render_mermaid_node(node, result):
    node_id = hash(node)
    pretty_name = " ".join(
        map(_clean_for_mermaid_name, [node, result.result, result.state])
    )[:100]
    return f'{node_id}("{pretty_name}")'


def _render_mermaid_edge(edge) -> str:
    if edge.source:
        return f"{hash(edge.source)} --{visualization.get_edge_label(edge)}--> {hash(edge.destination)}"
    return gamla.pipe(
        edge.args,
        gamla.map(
            lambda source: f"{hash(source)} --{visualization.get_edge_label(edge)}--> {hash(edge.destination)}"
        ),
        "\n".join,
    )


def _is_edge_participating(in_trace_nodes):
    def is_edge_participating(edge):
        return in_trace_nodes(edge.destination) and gamla.anymap(in_trace_nodes)(
            [edge.source, *edge.args]
        )

    return is_edge_participating


def mermaid_computation_trace(graph_instance: base_types.GraphType):
    def mermaid_computation_trace(node_to_results: run.NodeToResults):
        gamla.pipe(
            _node_computation_trace(
                node_to_results, graph.infer_graph_sink(graph_instance)
            ),
            lambda trace: [
                "",  # This is to avoid the indent of the logging details.
                "graph TD",
                gamla.pipe(trace, gamla.map(_render_mermaid_node), "\n".join),
                gamla.pipe(
                    graph_instance,
                    gamla.filter(
                        _is_edge_participating(
                            gamla.contains(frozenset(map(gamla.head, trace)))
                        )
                    ),
                    gamla.map(_render_mermaid_edge),
                    "\n".join,
                ),
            ],
            "\n".join,
            logging.info,
        )

    return mermaid_computation_trace
