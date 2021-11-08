from __future__ import annotations

from typing import Any, Callable, Optional, Text, Union

import gamla
from gamla.optimized import sync as opt_gamla

from computation_graph import base_types, graph

_ComposersInputType = Union[Callable, base_types.ComputationNode, base_types.GraphType]

_ComputationNodeOrGraphType = Union[base_types.ComputationNode, base_types.GraphType]


class _ComputationError:
    pass


_callable_or_graph_type_to_node_or_graph_type = gamla.unless(
    gamla.is_instance(tuple), graph.make_computation_node
)


def _get_edges_from_node_or_graph(
    node_or_graph: _ComputationNodeOrGraphType,
) -> base_types.GraphType:
    if isinstance(node_or_graph, base_types.ComputationNode):
        return ()
    return node_or_graph


@gamla.curry
def _get_unbound_signature_for_single_node(
    node_to_incoming_edges, node: base_types.ComputationNode
) -> base_types.NodeSignature:
    """Computes the new signature of unbound variables after considering internal edges."""
    incoming_edges = node_to_incoming_edges(node)

    keep_not_in_bound_kwargs = gamla.pipe(
        incoming_edges,
        opt_gamla.map(base_types.edge_key),
        gamla.filter(gamla.identity),
        frozenset,
        gamla.contains,
        gamla.remove,
    )

    return base_types.NodeSignature(
        is_args=node.signature.is_args
        and not any(edge.args for edge in incoming_edges),
        kwargs=tuple(keep_not_in_bound_kwargs(node.signature.kwargs)),
        optional_kwargs=tuple(keep_not_in_bound_kwargs(node.signature.optional_kwargs)),
    )


@gamla.curry
def make_optional(
    func: _ComposersInputType, default_value: Any
) -> base_types.GraphType:
    def return_default_value():
        return default_value

    return make_first(func, graph.make_computation_node(return_default_value))


def make_and(funcs, merge_fn: Callable) -> base_types.GraphType:
    def args_to_tuple(*args):
        return args

    merge_node = graph.make_computation_node(args_to_tuple)

    return gamla.pipe(
        funcs,
        gamla.map(_callable_or_graph_type_to_node_or_graph_type),
        tuple,
        gamla.juxtcat(
            gamla.mapcat(_get_edges_from_node_or_graph),
            gamla.compose_left(
                gamla.map(_infer_sink),
                tuple,
                lambda nodes: (
                    graph.make_standard_edge(source=nodes, destination=merge_node),
                    graph.make_standard_edge(
                        source=merge_node,
                        destination=graph.make_computation_node(merge_fn),
                        key="args",
                    ),
                ),
            ),
        ),
        tuple,
    )


def make_or(funcs, merge_fn: Callable) -> base_types.GraphType:
    def filter_computation_errors(*args):
        return gamla.pipe(
            args, gamla.remove(gamla.is_instance(_ComputationError)), tuple
        )

    filter_node = graph.make_computation_node(filter_computation_errors)
    merge_node = graph.make_computation_node(merge_fn)
    return gamla.pipe(
        funcs,
        gamla.map(make_optional(default_value=_ComputationError())),
        tuple,
        gamla.juxtcat(
            gamla.concat,
            gamla.compose_left(
                gamla.map(_infer_sink),
                tuple,
                lambda sinks: (
                    graph.make_standard_edge(source=sinks, destination=filter_node),
                    graph.make_standard_edge(
                        source=filter_node, destination=merge_node, key="args"
                    ),
                ),
            ),
        ),
        tuple,
    )


def _infer_sink(
    graph_or_node: Union[_ComputationNodeOrGraphType],
) -> base_types.ComputationNode:
    if isinstance(graph_or_node, base_types.ComputationNode):
        return graph_or_node
    return graph.infer_graph_sink_excluding_terminals(graph_or_node)


def _add_first_edge(
    source: _ComputationNodeOrGraphType,
    destination: base_types.ComputationNode,
    key: Text,
    priority: int,
) -> base_types.GraphType:
    return (
        graph.make_edge(
            is_future=False,
            priority=priority,
            source=_infer_sink(source),
            destination=destination,
            key=key,
        ),
        *_get_edges_from_node_or_graph(source),
    )


def make_first(*funcs: _ComposersInputType) -> base_types.GraphType:
    def first(first_input):
        return first_input

    assert funcs, "Expected at least one function."

    first_node = graph.make_computation_node(first)

    return gamla.pipe(
        funcs,
        gamla.map(_callable_or_graph_type_to_node_or_graph_type),
        enumerate,
        gamla.mapcat(
            gamla.star(
                lambda priority, node: _add_first_edge(
                    destination=first_node,
                    key="first_input",
                    priority=priority,
                    source=node,
                )
            )
        ),
        tuple,
    )


@gamla.curry
def _infer_composition_edges(
    key: Optional[Text],
    source: _ComputationNodeOrGraphType,
    destination: _ComputationNodeOrGraphType,
) -> base_types.GraphType:
    if isinstance(destination, base_types.ComputationNode):
        assert (
            key is None or key in destination.signature.kwargs
        ), f"Cannot compose, destination signature does not contain key '{key}'"

        return (
            graph.make_standard_edge(
                source=_infer_sink(source), destination=destination, key=key
            ),
            *_get_edges_from_node_or_graph(source),
        )

    return (
        gamla.pipe(
            destination,
            gamla.mapcat(graph.get_edge_nodes),
            gamla.unique,
            gamla.filter(
                gamla.compose_left(
                    _get_unbound_signature_for_single_node(
                        graph.get_incoming_edges_for_node(destination)
                    ),
                    lambda signature: key in signature.kwargs,
                )
            ),
            # Do not add edges to nodes from source that are already present in destination (cycle).
            gamla.filter(
                lambda node: isinstance(source, base_types.ComputationNode)
                or node not in graph.get_all_nodes(source)
            ),
            gamla.map(
                lambda node: graph.make_standard_edge(
                    source=_infer_sink(source), destination=node, key=key
                )
            ),
            tuple,
            gamla.check(
                gamla.identity,
                AssertionError(
                    f"Cannot compose, destination signature does not contain key '{key}'"
                ),
            ),
        )
        + destination
        + _get_edges_from_node_or_graph(source)
    )


def make_compose(
    *funcs: _ComposersInputType, key: Optional[Text] = None
) -> base_types.GraphType:
    assert (
        len(funcs) > 1
    ), f"Only {len(funcs)} function passed to compose, need at least 2, funcs={funcs}"
    return gamla.pipe(
        funcs,
        reversed,
        gamla.map(_callable_or_graph_type_to_node_or_graph_type),
        gamla.sliding_window(2),
        gamla.mapcat(gamla.star(_infer_composition_edges(key))),
        tuple,
    )
