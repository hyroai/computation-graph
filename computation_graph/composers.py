from __future__ import annotations

from typing import Any, Callable, Optional, Text, Tuple, Union

import gamla
import toolz
from toolz import curried

from computation_graph import base_types, graph

_ComposersInputType = Union[Callable, base_types.ComputationNode, base_types.GraphType]

_ComputationNodeOrGraphType = Union[base_types.ComputationNode, base_types.GraphType]


class _ComputationError:
    pass


_callable_or_graph_type_to_node_or_graph_type = gamla.curried_ternary(
    lambda x: isinstance(x, tuple), toolz.identity, graph.make_computation_node
)


def _get_edges_from_node_or_graph(
    node_or_graph: _ComputationNodeOrGraphType,
) -> base_types.GraphType:
    if isinstance(node_or_graph, base_types.ComputationNode):
        return ()
    return node_or_graph


def _signature_union(
    sig_a: base_types.NodeSignature, sig_b: base_types.NodeSignature
) -> base_types.NodeSignature:
    return base_types.NodeSignature(
        is_args=sig_a.is_args or sig_b.is_args,
        # Union signature does not preserve arguments' order.
        kwargs=tuple(set(sig_a.kwargs + sig_b.kwargs)),
        optional_kwargs=tuple(set(sig_a.optional_kwargs + sig_b.optional_kwargs)),
    )


@toolz.curry
def _get_unbound_signature_for_single_node(
    node: base_types.ComputationNode, edges: base_types.GraphType,
) -> base_types.NodeSignature:
    """Computes the new signature of unbound variables after considering internal edges."""
    incoming_edges = graph.get_incoming_edges_for_node(edges, node)

    bound_kwargs: Tuple[Text, ...] = toolz.pipe(
        incoming_edges, curried.map(lambda edge: edge.key), curried.filter(None)
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
    return toolz.pipe(
        edges,
        graph.get_all_nodes,
        curried.map(_get_unbound_signature_for_single_node(edges=edges)),
        curried.reduce(_signature_union),
    )


@toolz.curry
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

    return toolz.pipe(
        funcs,
        curried.map(_callable_or_graph_type_to_node_or_graph_type),
        tuple,
        gamla.juxtcat(
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
        tuple,
    )


def make_or(funcs, merge_fn: Callable) -> base_types.GraphType:
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
        curried.map(make_optional(default_value=_ComputationError())),
        tuple,
        gamla.juxtcat(
            curried.concat,
            toolz.compose_left(
                curried.map(_infer_sink),
                tuple,
                lambda sinks: (
                    graph.make_edge(source=sinks, destination=filter_node,),
                    graph.make_edge(
                        source=filter_node, destination=merge_node, key="args",
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
    return graph.infer_graph_sink(graph_or_node)


def _add_first_edge(
    source: _ComputationNodeOrGraphType,
    destination: base_types.ComputationNode,
    key: Text,
    priority: int,
) -> base_types.GraphType:
    return (
        graph.make_edge(
            source=_infer_sink(source),
            destination=destination,
            key=key,
            priority=priority,
        ),
        *_get_edges_from_node_or_graph(source),
    )


def make_first(*funcs: _ComposersInputType) -> base_types.GraphType:
    def first(first_input):
        return first_input

    assert funcs, "Expected at least one function."

    first_node = graph.make_computation_node(first)

    return toolz.pipe(
        funcs,
        curried.map(_callable_or_graph_type_to_node_or_graph_type),
        enumerate,
        curried.mapcat(
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


@toolz.curry
def _infer_composition_edges(
    source: _ComputationNodeOrGraphType,
    destination: _ComputationNodeOrGraphType,
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
            *_get_edges_from_node_or_graph(source),
        )

    return (
        toolz.pipe(
            destination,
            curried.mapcat(graph.get_edge_nodes),
            curried.unique,
            curried.filter(
                toolz.compose_left(
                    _get_unbound_signature_for_single_node(edges=destination),
                    lambda signature: key in signature.kwargs,
                )
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
        curried.mapcat(gamla.star(_infer_composition_edges(key=key))),
        tuple,
    )
