from typing import Callable, Text, Tuple
from xml.sax import saxutils

import gamla
import pygraphviz as pgv
import toolz
from toolz import curried

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


def _add_computation_node(
    pgv_graph: pgv.AGraph,
    node: base_types.ComputationNode,
):
    node_id = hash(node)
    pgv_graph.add_node(
        node_id,
        label=saxutils.quoteattr(str(node)),
        shape=_get_node_shape(node),
    )

    if node.is_stateful:
        pgv_graph.add_edge(node_id, node_id)


def computation_graph_to_graphviz(edges: base_types.GraphType) -> pgv.AGraph:
    pgv_graph = pgv.AGraph(directed=True)
    for edge in edges:
        _add_computation_node(pgv_graph, edge.destination)
        if edge.source:
            _add_computation_node(pgv_graph, edge.source)
            pgv_graph.add_edge(
                hash(edge.source),
                hash(edge.destination),
                label=_get_edge_label(edge),
            )
        else:
            for source in edge.args:
                _add_computation_node(pgv_graph, source)
                pgv_graph.add_edge(
                    hash(source),
                    hash(edge.destination),
                    label=_get_edge_label(edge),
                )

    return pgv_graph


def _do_add_edge(result_graph: pgv.AGraph) -> Callable[[pgv.Edge], pgv.AGraph]:
    return lambda edge: toolz.pipe(
        result_graph,
        curried.do(lambda g: g.add_edge(edge, **edge.attr)),
    )


def _do_add_node(result_graph: pgv.AGraph) -> Callable[[pgv.Node], pgv.AGraph]:
    return lambda node: toolz.pipe(
        result_graph,
        curried.do(lambda g: g.add_node(node, **node.attr)),
    )


def union_graphviz(graphs: Tuple[pgv.AGraph, ...]) -> pgv.AGraph:
    return toolz.pipe(
        graphs,
        # copy to avoid side effects that influence caller
        gamla.juxt(gamla.compose_left(toolz.first, pgv.AGraph.copy), curried.drop(1)),
        gamla.star(
            gamla.reduce(
                lambda result_graph, another_graph: toolz.pipe(
                    another_graph,
                    # following does side effects on result_graph, that's why we return just(result_graph)
                    # we assume no parallelization in gamla.juxt
                    curried.do(
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
                        ),
                    ),
                    gamla.just(result_graph),
                ),
            ),
        ),
    )


def save_graphviz_as_png(filename: Text) -> Callable[[pgv.AGraph], pgv.AGraph]:
    return curried.do(lambda pgv_graph: pgv_graph.draw(filename, prog="dot"))


def save_graphviz_as_dot(filename: Text) -> Callable[[pgv.AGraph], pgv.AGraph]:
    return curried.do(lambda pgv_graph: pgv_graph.write(filename))


@toolz.curry
def computation_trace_to_graphviz(
    computation_trace: Tuple[
        Tuple[base_types.ComputationNode, base_types.ComputationResult]
    ],
) -> pgv.AGraph:
    pgv_graph = pgv.AGraph()
    for (node, result) in computation_trace:
        _add_computation_node(pgv_graph, node)
        graphviz_node = pgv_graph.get_node(hash(node))
        graphviz_node.attr["color"] = "red"
        graphviz_node.attr["result"] = str(
            result.result,
        )[:200]
        graphviz_node.attr["state"] = str(
            result.state,
        )[:200]

    return pgv_graph


visualize_graph = gamla.compose_left(
    computation_graph_to_graphviz,
    save_graphviz_as_png("topology.png"),
)
serialize_computation_trace = toolz.compose_left(
    save_graphviz_as_dot,
    gamla.before(
        gamla.compose_left(
            gamla.stack([computation_graph_to_graphviz, computation_trace_to_graphviz]),
            union_graphviz,
            curried.do(lambda g: g.layout(prog="dot")),
        ),
    ),
)
