import logging

import gamla

from computation_graph import base_types
from computation_graph.trace import trace_utils


def _clean_for_mermaid_name(obj):
    return str(obj).replace('"', "")


@gamla.star
def _render_mermaid_node(node, result):
    node_id = hash(node)
    pretty_name = " ".join(map(_clean_for_mermaid_name, [node, result]))[:100]
    return f'{node_id}("{pretty_name}")'


def _render_mermaid_edge(edge) -> str:
    if edge.source:
        return f"{hash(edge.source)} --{trace_utils.get_edge_label(edge)}--> {hash(edge.destination)}"
    return gamla.pipe(
        edge.args,
        gamla.map(
            lambda source: f"{hash(source)} --{trace_utils.get_edge_label(edge)}--> {hash(edge.destination)}"
        ),
        "\n".join,
    )


def mermaid_computation_trace(graph_instance: base_types.GraphType):
    return gamla.compose_left(
        dict.items,
        frozenset,
        lambda trace: [
            "",  # This is to avoid the indent of the logging details.
            "graph TD",
            gamla.pipe(trace, gamla.map(_render_mermaid_node), "\n".join),
            gamla.pipe(
                graph_instance,
                gamla.filter(
                    trace_utils.is_edge_participating(
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
