from typing import Callable, Iterable, Tuple
from xml.sax import saxutils

import gamla
import pygraphviz as pgv

from computation_graph import base_types, graph
from computation_graph.trace import trace_utils


def _get_node_shape(node: base_types.ComputationNode):
    if node.name == "first":
        return "triangle"
    return "ellipse"


def _add_computation_node(pgv_graph: pgv.AGraph, node: base_types.ComputationNode):
    node_id = hash(node)
    pgv_graph.add_node(
        node_id, label=saxutils.quoteattr(str(node)), shape=_get_node_shape(node)
    )
    if node.is_stateful:
        pgv_graph.add_edge(node_id, node_id)


def _handle_edge(pgv_graph, edge):
    _add_computation_node(pgv_graph, edge.destination)
    if edge.source:
        _add_computation_node(pgv_graph, edge.source)
        pgv_graph.add_edge(
            hash(edge.source),
            hash(edge.destination),
            label=trace_utils.get_edge_label(edge),
            style="dashed" if edge.is_future else "",
        )
    else:
        for source in edge.args:
            _add_computation_node(pgv_graph, source)
            pgv_graph.add_edge(
                hash(source),
                hash(edge.destination),
                label=trace_utils.get_edge_label(edge),
                style="dashed" if edge.is_future else "",
            )


def computation_graph_to_graphviz(edges: base_types.GraphType) -> pgv.AGraph:
    pgv_graph = pgv.AGraph(directed=True)
    for edge in edges:
        _handle_edge(pgv_graph, edge)
    return pgv_graph


def _do_add_edge(result_graph: pgv.AGraph) -> Callable[[pgv.Edge], pgv.AGraph]:
    return lambda edge: gamla.pipe(
        result_graph, gamla.side_effect(lambda g: g.add_edge(edge, **edge.attr))
    )


def _do_add_node(result_graph: pgv.AGraph) -> Callable[[pgv.Node], pgv.AGraph]:
    return lambda node: gamla.pipe(
        result_graph, gamla.side_effect(lambda g: g.add_node(node, **node.attr))
    )


union_graphviz: Callable[[Iterable[pgv.AGraph]], pgv.AGraph] = gamla.compose_left(
    # copy to avoid side effects that influence caller
    gamla.juxt(gamla.compose_left(gamla.head, pgv.AGraph.copy), gamla.drop(1)),
    gamla.star(
        gamla.reduce(
            lambda result_graph, another_graph: gamla.pipe(
                another_graph,
                # following does side effects on result_graph, that's why we return just(result_graph)
                # we assume no parallelization in gamla.juxt
                gamla.side_effect(
                    gamla.juxt(
                        gamla.compose_left(
                            pgv.AGraph.nodes,
                            gamla.map(_do_add_node(result_graph)),
                            tuple,
                        ),
                        gamla.compose_left(
                            pgv.AGraph.edges,
                            gamla.map(_do_add_edge(result_graph)),
                            tuple,
                        ),
                    )
                ),
                gamla.just(result_graph),
            )
        )
    ),
)


def _save_as_png(filename: str) -> Callable[[pgv.AGraph], pgv.AGraph]:
    return gamla.side_effect(lambda pgv_graph: pgv_graph.write(filename))


def computation_trace_to_graphviz(
    computation_trace: Iterable[
        Tuple[base_types.ComputationNode, base_types.ComputationResult]
    ]
) -> pgv.AGraph:
    pgv_graph = pgv.AGraph()
    for node, result in computation_trace:
        _add_computation_node(pgv_graph, node)
        graphviz_node = pgv_graph.get_node(hash(node))
        graphviz_node.attr["color"] = "red"
        graphviz_node.attr["result"] = str(result.result)[:200]
        graphviz_node.attr["state"] = str(result.state)[:200]

    return pgv_graph


visualize_graph = gamla.compose_left(
    computation_graph_to_graphviz, _save_as_png("bot_computation_graph.dot")
)


@gamla.curry
def computation_trace(
    filename: str, graph_instance: base_types.GraphType, node_to_results: Callable
):
    gviz = union_graphviz(
        [
            computation_graph_to_graphviz(graph_instance),
            computation_trace_to_graphviz(
                trace_utils.node_computation_trace(
                    node_to_results, graph.infer_graph_sink(graph_instance)
                )
            ),
        ]
    )
    gviz.layout(prog="dot")
    gviz.write(filename)
