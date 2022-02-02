import pprint
from typing import Callable, Iterable, Tuple

import gamla

from computation_graph import base_types
from computation_graph.trace import trace_utils

_NodeTree = Tuple[base_types.ComputationNode, Tuple["_NodeTree", ...]]  # type: ignore
_NodeAndResultTree = Tuple[  # type: ignore
    base_types.ComputationNode,
    base_types.Result,
    Tuple["_NodeAndResultTree", ...],  # type: ignore
]


@gamla.curry
def _process_node(
    node_to_result: Callable[[_NodeTree], base_types.Result],
    source_and_destination_to_edges: Callable[
        [Tuple[base_types.ComputationNode, base_types.ComputationNode]],
        Iterable[base_types.ComputationEdge],
    ],
    node: _NodeTree,
    children: Iterable[_NodeAndResultTree],
) -> _NodeAndResultTree:
    return (
        node[0],
        node_to_result(node),
        gamla.pipe(
            children,
            gamla.map(
                gamla.juxt(
                    gamla.compose_left(
                        gamla.head,
                        gamla.pair_right(gamla.just(node[0])),
                        source_and_destination_to_edges,
                        # In theory there can be >1 connections between two nodes.
                        gamla.map(base_types.edge_key),
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
def _skip_uninsteresting_nodes(node_to_result, node, children) -> _NodeTree:
    """To make the debug log more readable, we try to reduce uninteresting steps by some heuristics."""
    result = node_to_result(node)
    children = tuple(children)
    if len(children) == 1:
        first_child = gamla.head(children)
        if not _should_render(result):
            return first_child
        if result == node_to_result(children[0][0]):
            return first_child
    return node, children


_index_by_destination = gamla.compose_left(
    gamla.groupby(base_types.edge_destination), gamla.dict_to_getter_with_default(())
)


def _edge_to_node_pairs(edge: base_types.ComputationEdge):
    if edge.source:
        return [(edge.source, edge.destination)]
    return gamla.pipe((edge.args, edge.destination), gamla.explode(0))


_index_by_source_and_destination = gamla.compose_left(
    gamla.groupby_many(_edge_to_node_pairs), gamla.dict_to_getter_with_default(())
)

_sources = gamla.compose(frozenset, gamla.mapcat(base_types.edge_sources))


@gamla.curry
def _trace_single_output(
    source_and_destination_to_edges,
    destination_to_edges,
    node_has_result,
    node_to_result: Callable,
):
    return gamla.compose_left(
        gamla.tree_reduce(
            gamla.compose_left(
                destination_to_edges,
                gamla.filter(trace_utils.is_edge_participating(node_has_result)),
                gamla.mapcat(base_types.edge_sources),
            ),
            _skip_uninsteresting_nodes(node_to_result),
        ),
        gamla.tree_reduce(
            gamla.nth(1),
            _process_node(
                gamla.compose_left(gamla.head, node_to_result),
                source_and_destination_to_edges,
            ),
        ),
    )


def computation_trace(graph_instance: base_types.GraphType) -> _NodeAndResultTree:
    regular_edges = gamla.pipe(
        graph_instance, gamla.remove(base_types.edge_is_future), frozenset
    )
    sinks = gamla.pipe(
        regular_edges,
        gamla.map(base_types.edge_destination),
        gamla.remove(gamla.contains(_sources(regular_edges))),
        frozenset,
    )

    return gamla.compose(
        pprint.pprint,
        lambda trace_single_node: gamla.pipe(
            sinks, gamla.map(trace_single_node), tuple
        ),
        gamla.star(
            _trace_single_output(
                _index_by_source_and_destination(graph_instance),
                _index_by_destination(graph_instance),
            )
        ),
        # We set a default here so we don't get a `KeyError` for nodes that could not be computed.
        gamla.juxt(gamla.contains, gamla.dict_to_getter_with_default(None)),
    )
