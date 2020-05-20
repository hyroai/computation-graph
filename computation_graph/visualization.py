from typing import Text

import pygraphviz as pgv

from computation_graph import base_types, graph


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


def _add_single_node(
    pgv_graph: pgv.AGraph,
    edges: base_types.GraphType,
    node: base_types.ComputationNode,
):
    node_id = graph.infer_node_id(edges, node)
    pgv_graph.add_node(node_id, label=node, shape=_get_node_shape(node))

    if node.is_stateful:
        pgv_graph.add_edge(node_id, node_id)


def _make_graph_from_edges(pgv_graph: pgv.AGraph, edges: base_types.GraphType):
    for edge in edges:
        _add_single_node(pgv_graph, edges, edge.destination)
        if edge.source:
            _add_single_node(pgv_graph, edges, edge.source)
            pgv_graph.add_edge(
                graph.infer_node_id(edges, edge.source),
                graph.infer_node_id(edges, edge.destination),
                label=_get_edge_label(edge),
            )
        else:
            for source in edge.args:
                _add_single_node(pgv_graph, edges, source)
                pgv_graph.add_edge(
                    graph.infer_node_id(edges, source),
                    graph.infer_node_id(edges, edge.destination),
                    label=_get_edge_label(edge),
                )


def visualize_graph(edges: base_types.GraphType, filename: Text = "topology.png"):
    pgv_graph = pgv.AGraph(directed=True)
    _make_graph_from_edges(pgv_graph, edges)
    pgv_graph.draw(filename, prog="dot")
