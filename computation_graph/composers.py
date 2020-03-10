from __future__ import annotations

import dataclasses
import functools
from typing import Any, Callable, Optional, Text, Tuple, Type, Union

import gamla
import toolz
from toolz import curried

from computation_graph import base_types, graph

_ComposersInputType = Union[Callable, base_types.ComputationNode, base_types.GraphType]

_ComputationNodeAndGraphType = Union[base_types.ComputationNode, base_types.GraphType]


class _ComputationError:
    pass


_callable_or_graph_type_to_node_or_graph_type = gamla.curried_ternary(
    lambda x: isinstance(x, tuple), toolz.identity, graph.make_computation_node
)


def _get_edges_from_node_or_graph(
    node_or_graph: _ComputationNodeAndGraphType,
) -> base_types.GraphType:
    return (
        ()  # type:ignore
        if isinstance(node_or_graph, base_types.ComputationNode)
        else node_or_graph
    )


def _signature_union(
    sig_a: base_types.NodeSignature, sig_b: base_types.NodeSignature
) -> base_types.NodeSignature:
    return base_types.NodeSignature(
        is_args=sig_a.is_args or sig_b.is_args,
        # Union signature does not preserve arguments' order.
        kwargs=tuple(set(sig_a.kwargs + sig_b.kwargs)),
        optional_kwargs=tuple(set(sig_a.optional_kwargs + sig_b.optional_kwargs)),
    )


def _get_unbound_signature_for_single_node(
    edges: base_types.GraphType, node: base_types.ComputationNode
) -> base_types.NodeSignature:
    """Computes the new signature of unbound variables after considering internal edges."""
    incoming_edges = graph.get_incoming_edges_for_node(edges, node)

    bound_kwargs: Tuple[Text, ...] = tuple(
        filter(None, map(lambda edge: edge.key, incoming_edges))
    )

    return base_types.NodeSignature(
        is_args=node.signature.is_args
        and not any(edge.args for edge in incoming_edges),
        kwargs=tuple(filter(lambda x: x not in bound_kwargs, node.signature.kwargs)),
        optional_kwargs=tuple(
            filter(lambda x: x not in bound_kwargs, node.signature.optional_kwargs)
        ),
    )


def _get_unbound_signature_for_graph(
    edges: base_types.GraphType,
) -> base_types.NodeSignature:
    return functools.reduce(
        _signature_union,
        map(
            lambda x: _get_unbound_signature_for_single_node(edges, x),
            graph.get_all_nodes(edges),
        ),
    )


@toolz.curry
def make_optional(
    func: _ComposersInputType, exception_type: Type[Exception], default_value: Any
) -> base_types.GraphType:
    def return_default_value():
        return default_value

    return make_first(
        func,
        graph.make_computation_node(return_default_value),
        exception_type=exception_type,
    )


def make_and(funcs, merge_fn: Callable) -> base_types.GraphType:
    def args_to_tuple(*args):
        return args

    merge_node = graph.make_computation_node(args_to_tuple)

    return toolz.pipe(
        funcs,
        curried.map(_callable_or_graph_type_to_node_or_graph_type),
        tuple,
        curried.juxt(
            curried.mapcat(_get_edges_from_node_or_graph),
            curried.compose_left(
                curried.map(_infer_sink),
                tuple,
                lambda nodes: (
                    graph.make_edge(source=nodes, destination=merge_node),
                    graph.make_edge(
                        source=merge_node,
                        destination=graph.make_computation_node(merge_fn),
                        key="args",
                    ),
                ),
            ),
        ),
        curried.concat,
        tuple,
    )


def make_or(
    funcs, merge_fn: Callable, exception_type: Type[Exception]
) -> base_types.GraphType:
    def filter_computation_errors(*args):
        return toolz.pipe(
            args, curried.filter(lambda x: not isinstance(x, _ComputationError)), tuple
        )

    def identity(x):
        return x

    filter_node = graph.make_computation_node(filter_computation_errors)
    merge_node = graph.make_computation_node(merge_fn)
    return toolz.pipe(
        funcs,
        curried.map(
            make_optional(
                exception_type=exception_type, default_value=_ComputationError()
            )
        ),
        tuple,
        curried.juxt(
            curried.concat,
            toolz.compose_left(
                curried.map(_infer_sink),
                tuple,
                lambda sinks: (
                    graph.make_edge(
                        source=sinks,
                        destination=filter_node,
                        allowed_exceptions=frozenset([exception_type]),
                    ),
                    graph.make_edge(
                        source=filter_node, destination=merge_node, key="args",
                    ),
                    # Add an edge to catch handled exceptions when running `merge_fn`.
                    graph.make_edge(
                        source=merge_node,
                        destination=graph.make_computation_node(identity),
                        allowed_exceptions=frozenset([exception_type]),
                    ),
                ),
            ),
        ),
        curried.concat,
        tuple,
    )


def _infer_sink(
    graph_or_node: Union[_ComputationNodeAndGraphType],
) -> base_types.ComputationNode:
    if isinstance(graph_or_node, base_types.ComputationNode):
        return graph_or_node
    return graph.infer_graph_sink(graph_or_node)


def _add_first_edge(
    source: _ComputationNodeAndGraphType,
    destination: base_types.ComputationNode,
    key: Text,
    priority: int,
    exception_type: Type[Exception],
) -> base_types.GraphType:
    return (
        graph.make_edge(
            source=_infer_sink(source),
            destination=destination,
            key=key,
            priority=priority,
            allowed_exceptions=frozenset([exception_type]),
        ),
    ) + toolz.pipe(
        _get_edges_from_node_or_graph(source),
        curried.map(
            lambda edge: dataclasses.replace(
                edge,
                allowed_exceptions=edge.allowed_exceptions
                | frozenset([exception_type]),
            )
        ),
        tuple,
    )


def make_first(
    *funcs: _ComposersInputType, exception_type: Type[Exception]
) -> base_types.GraphType:
    def first(first_input):
        return first_input

    assert funcs, "cannot use first on empty list of funcs"

    first_node = graph.make_computation_node(first)

    return toolz.pipe(
        funcs,
        curried.map(_callable_or_graph_type_to_node_or_graph_type),
        enumerate,
        curried.mapcat(
            lambda node_and_priority: _add_first_edge(
                destination=first_node,
                key="first_input",
                priority=toolz.first(node_and_priority),
                source=toolz.second(node_and_priority),
                exception_type=exception_type,
            )
        ),
        tuple,
    )


@toolz.curry
def _infer_composition_edges(
    source: _ComputationNodeAndGraphType,
    destination: _ComputationNodeAndGraphType,
    key: Optional[Text] = None,
) -> base_types.GraphType:
    if isinstance(destination, base_types.ComputationNode):
        assert (
            key is None or key in destination.signature.kwargs
        ), f"Cannot compose, destination signature does not contain key '{key}'"

        return (
            graph.make_edge(
                source=_infer_sink(source), destination=destination, key=key
            ),
        ) + _get_edges_from_node_or_graph(source)

    return (
        toolz.pipe(
            destination,
            curried.mapcat(graph.get_edge_nodes),
            curried.unique,
            curried.filter(
                lambda node: key
                in _get_unbound_signature_for_single_node(
                    edges=destination, node=node
                ).kwargs
            ),
            # Do not add edges to nodes from source that are already present in destination (cycle).
            curried.filter(
                lambda node: isinstance(source, base_types.ComputationNode)
                or node not in graph.get_all_nodes(source)
            ),
            curried.map(
                lambda node: graph.make_edge(
                    source=_infer_sink(source), destination=node, key=key
                )
            ),
            tuple,
            gamla.check(
                toolz.identity,
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
    return toolz.pipe(
        funcs,
        reversed,
        curried.map(_callable_or_graph_type_to_node_or_graph_type),
        curried.sliding_window(2),
        curried.mapcat(
            gamla.star(
                lambda source, destination: _infer_composition_edges(
                    source=source, destination=destination, key=key
                )
            )
        ),
        tuple,
    )
