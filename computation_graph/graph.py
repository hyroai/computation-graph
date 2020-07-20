import functools
import inspect
from typing import Callable, FrozenSet, Optional, Text, Tuple, Union

import gamla
import toolz
from toolz import curried

from computation_graph import base_types


def get_all_nodes(
    edges: base_types.GraphType,
) -> FrozenSet[base_types.ComputationNode]:
    return toolz.pipe(edges, curried.mapcat(get_edge_nodes), toolz.unique, frozenset)


def _infer_callable_signature(func: Callable) -> base_types.NodeSignature:
    signature = base_types.NodeSignature(
        is_args=any(
            "*" + x.name == str(x)
            for x in tuple(inspect.signature(func).parameters.values())
        ),
        kwargs=tuple(
            x.name
            for x in tuple(inspect.signature(func).parameters.values())
            if "*" + x.name != str(x)
        ),
        optional_kwargs=tuple(
            x.name
            for x in tuple(inspect.signature(func).parameters.values())
            if "*" + x.name != str(x) and x.default != x.empty
        ),
    )

    assert not any(
        "**" + x.name == str(x)
        for x in tuple(inspect.signature(func).parameters.values())
    ), f"Cannot infer signature for {func}, signature includes **kwargs"

    return signature


def _infer_callable_name(func: Callable) -> Text:
    if isinstance(func, functools.partial):
        return func.func.__name__
    return func.__name__


get_edge_nodes = functools.lru_cache(maxsize=1024)(
    gamla.curried_ternary(
        lambda edge: edge.args,
        lambda edge: edge.args + (edge.destination,),
        lambda edge: (edge.source, edge.destination),
    ),
)


_edges_to_node_map = toolz.compose_left(
    curried.mapcat(get_edge_nodes), curried.unique, enumerate, dict,
)


@toolz.curry
def infer_node_id(edges: base_types.GraphType, node: base_types.ComputationNode) -> int:
    return toolz.pipe(edges, _edges_to_node_map, curried.itemmap(reversed))[node]


@toolz.curry
def infer_node_from_id(
    edges: base_types.GraphType, node_id: int,
) -> base_types.ComputationNode:
    return _edges_to_node_map(edges)[node_id]


def make_computation_node(
    func: Union[base_types.ComputationNode, Callable],
) -> base_types.ComputationNode:
    assert func is not base_types.ComputationNode
    if isinstance(func, base_types.ComputationNode):
        return func

    return base_types.ComputationNode(
        name=_infer_callable_name(func),
        func=func,
        signature=_infer_callable_signature(func),
    )


@toolz.curry
def make_edge(
    priority: int,
    is_future: bool,
    source: Union[
        Callable,
        Tuple[Union[Callable, base_types.ComputationNode], ...],
        base_types.ComputationNode,
    ],
    destination: Union[Callable, base_types.ComputationNode],
    key: Optional[Text],
) -> base_types.ComputationEdge:
    assert (
        callable(source)
        or isinstance(source, base_types.ComputationNode)
        or isinstance(source, tuple)
    ), "cannot create edge: source must be a callable, ComputationNode or tuple"
    assert callable(destination) or isinstance(
        destination, base_types.ComputationNode,
    ), "cannot create edge: destination must be a callable or a ComputationNode"
    if isinstance(source, tuple):
        return base_types.ComputationEdge(
            args=tuple(map(make_computation_node, source)),
            source=None,
            destination=make_computation_node(destination),
            key=None,
            priority=priority,
            is_future=is_future,
        )

    return base_types.ComputationEdge(
        source=make_computation_node(source),
        destination=make_computation_node(destination),
        args=(),
        key=key,
        priority=priority,
        is_future=is_future,
    )


make_default_edge = make_edge(0, False)
make_future_edge = make_edge(0, True)


def _get_leaves(edges: base_types.GraphType) -> FrozenSet[base_types.ComputationNode]:
    return toolz.pipe(
        edges,
        curried.filter(lambda edge: not edge.is_future),
        get_all_nodes,
        curried.filter(
            lambda node: not any(
                edge.source == node or node in edge.args for edge in edges
            ),
        ),
        frozenset,
    )


def infer_graph_sink(edges: base_types.GraphType) -> base_types.ComputationNode:
    # TODO(nitzo): Handle:  f->g, g -> future -> h, f->h. infer sink should notify of an error here.
    return toolz.pipe(
        edges,
        # Check for special reducer case (x --> future_edge --> x).
        gamla.ternary(
            lambda edges: len(edges) == 1
            and edges[0].is_future
            and edges[0].source == edges[0].destination,
            gamla.just(frozenset([edges[0].destination])),
            _get_leaves,
        ),
        gamla.ternary(
            gamla.len_equals(1),
            toolz.identity,
            toolz.compose_left(
                lambda sinks: AssertionError(
                    f"computation graph must have exactly one sink {sinks}",
                ),
                gamla.make_raise,
                gamla.invoke,
            ),
        ),
        toolz.first,
    )


def get_incoming_edges_for_node(
    edges: base_types.GraphType, node: base_types.ComputationNode,
) -> FrozenSet[base_types.ComputationEdge]:
    return frozenset(filter(lambda edge: edge.destination == node, edges))


def get_outgoing_edges_for_node(
    edges: base_types.GraphType, node: base_types.ComputationNode,
) -> FrozenSet[base_types.ComputationEdge]:
    return toolz.pipe(
        edges,
        curried.filter(lambda edge: node == edge.source or node in edge.args),
        frozenset,
    )
