from typing import Callable, Text, Tuple
from xml.sax import saxutils

import gamla
import pygraphviz as pgv

from computation_graph import base_types


def _get_edge_label(edge: base_types.ComputationEdge):
    if edge.key in (None, "args"):
        return ""
    if edge.key == "first_input":
        return str(edge.priority)
    return edge.key or ""


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
            hash(edge.source), hash(edge.destination), label=_get_edge_label(edge)
        )
    else:
        for source in edge.args:
            _add_computation_node(pgv_graph, source)
            pgv_graph.add_edge(
                hash(source), hash(edge.destination), label=_get_edge_label(edge)
            )


def _computation_graph_to_graphviz(edges: base_types.GraphType) -> pgv.AGraph:
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


def _union_graphviz(graphs: Tuple[pgv.AGraph, ...]) -> pgv.AGraph:
    return gamla.pipe(
        graphs,
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


def _save_graphviz_as_png(filename: Text) -> Callable[[pgv.AGraph], pgv.AGraph]:
    return gamla.side_effect(lambda pgv_graph: pgv_graph.draw(filename, prog="dot"))


def _save_graphviz_as_dot(filename: Text) -> Callable[[pgv.AGraph], pgv.AGraph]:
    return gamla.side_effect(lambda pgv_graph: pgv_graph.write(filename))


def _computation_trace_to_graphviz(
    computation_trace: Tuple[
        Tuple[base_types.ComputationNode, base_types.ComputationResult]
    ]
) -> pgv.AGraph:
    pgv_graph = pgv.AGraph()
    for (node, result) in computation_trace:
        _add_computation_node(pgv_graph, node)
        graphviz_node = pgv_graph.get_node(hash(node))
        graphviz_node.attr["color"] = "red"
        graphviz_node.attr["result"] = str(result.result)[:200]
        graphviz_node.attr["state"] = str(result.state)[:200]

    return pgv_graph


visualize_graph = gamla.compose_left(
    _computation_graph_to_graphviz, _save_graphviz_as_png("topology.png")
)
serialize_computation_trace = gamla.compose_left(
    _save_graphviz_as_dot,
    gamla.before(
        gamla.compose_left(
            gamla.stack(
                [_computation_graph_to_graphviz, _computation_trace_to_graphviz]
            ),
            _union_graphviz,
            gamla.side_effect(lambda g: g.layout(prog="dot")),
        )
    ),
)
