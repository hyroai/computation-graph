import pprint
from typing import Callable

import gamla

from computation_graph import base_types, graph
from computation_graph.debug import trace_utils


@gamla.curry
def _process_node(node_to_result, source_and_destination_to_edges, node, children):
    return (
        node[0],
        node_to_result(node),
        gamla.pipe(
            children,
            gamla.map(
                gamla.juxt(
                    gamla.compose_left(
                        gamla.head,
                        lambda source_node: source_and_destination_to_edges(
                            (source_node, node)
                        ),
                        # In theory there can be >1 connections between two nodes.
                        gamla.map(gamla.attrgetter("key")),
                        frozenset,
                    ),
                    gamla.identity,
                )
            ),
            tuple,
        ),
    )


_should_render = gamla.compose_left(str, gamla.len_smaller(1000))


@gamla.curry
def _skip_uninsteresting_nodes(node_to_result, node, children):
    result = node_to_result(node)
    children = tuple(children)
    if len(children) == 1:
        first_child = gamla.head(children)
        if not _should_render(result):
            return first_child
        if result == node_to_result(children[0][0]):
            return first_child
    return (node, children)


_index_by_destination = gamla.compose_left(
    gamla.groupby(gamla.attrgetter("destination")),
    gamla.dict_to_getter_with_default(()),
)


def _edge_to_node_pairs(edge):
    if edge.source:
        return [(edge.source, edge.destination)]
    return gamla.pipe((edge.args, edge.destination), gamla.explode(0))


_index_by_source_and_destination = gamla.compose_left(
    gamla.groupby_many(_edge_to_node_pairs), gamla.dict_to_getter_with_default(())
)


def computation_trace(graph_instance: base_types.GraphType):
    destination_to_edges = _index_by_destination(graph_instance)
    source_and_destination_to_edges = _index_by_source_and_destination(graph_instance)

    def computation_trace(node_to_results: Callable):
        trace = trace_utils.node_computation_trace(
            node_to_results, graph.infer_graph_sink(graph_instance)
        )
        # So we don't get a `KeyError` in cases where the computation graph raises.
        node_to_result = gamla.dict_to_getter_with_default(None, dict(trace))
        gamla.pipe(
            graph_instance,
            graph.infer_graph_sink,
            gamla.tree_reduce(
                gamla.compose_left(
                    destination_to_edges,
                    gamla.filter(
                        trace_utils.is_edge_participating(
                            gamla.contains(frozenset(map(gamla.head, trace)))
                        )
                    ),
                    gamla.mapcat(
                        lambda edge: edge.args if edge.args else (edge.source,)
                    ),
                ),
                _skip_uninsteresting_nodes(node_to_result),
            ),
            gamla.tree_reduce(
                gamla.nth(1),
                _process_node(
                    gamla.compose_left(
                        gamla.head,
                        node_to_result,
                        gamla.unless(gamla.equals(None), gamla.attrgetter("result")),
                    ),
                    source_and_destination_to_edges,
                ),
            ),
            pprint.pprint,
        )

    return computation_trace
