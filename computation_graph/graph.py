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


def _is_reducer_type(node: Callable) -> bool:
    return "state" in inspect.signature(node).parameters


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
    )
)


_edges_to_node_id_map = toolz.compose_left(
    curried.mapcat(get_edge_nodes),
    curried.unique,
    enumerate,
    curried.map(reversed),
    dict,
)


@toolz.curry
def infer_node_id(edges: base_types.GraphType, node: base_types.ComputationNode) -> int:
    return _edges_to_node_id_map(edges)[node]


def make_computation_node(
    func: Union[base_types.ComputationNode, Callable]
) -> base_types.ComputationNode:
    assert func is not base_types.ComputationNode
    if isinstance(func, base_types.ComputationNode):
        return func

    return base_types.ComputationNode(
        name=_infer_callable_name(func),
        func=func,
        is_stateful=_is_reducer_type(func),
        signature=_infer_callable_signature(func),
    )


def make_edge(
    source: Union[
        Callable,
        Tuple[Union[Callable, base_types.ComputationNode], ...],
        base_types.ComputationNode,
    ],
    destination: Union[Callable, base_types.ComputationNode],
    key: Optional[Text] = None,
    priority: int = 0,
) -> base_types.ComputationEdge:
    if isinstance(source, tuple):
        return base_types.ComputationEdge(
            args=tuple(map(make_computation_node, source)),
            destination=make_computation_node(destination),
            priority=priority,
        )

    return base_types.ComputationEdge(
        source=make_computation_node(source),
        destination=make_computation_node(destination),
        key=key,
        priority=priority,
    )


def get_leaves(edges: base_types.GraphType) -> FrozenSet[base_types.ComputationNode]:
    return toolz.pipe(
        edges,
        get_all_nodes,
        curried.filter(
            lambda node: not any(
                edge.source == node or node in edge.args for edge in edges
            )
        ),
        frozenset,
    )


def infer_graph_sink(edges: base_types.GraphType) -> base_types.ComputationNode:
    return toolz.pipe(
        edges,
        get_leaves,
        gamla.check(
            gamla.len_equals(1),
            AssertionError("computation graph has more than one sink"),
        ),
        toolz.first,
    )


def get_incoming_edges_for_node(
    edges: base_types.GraphType, node: base_types.ComputationNode
) -> FrozenSet[base_types.ComputationEdge]:
    return frozenset(filter(lambda edge: edge.destination == node, edges))


def get_outgoing_edges_for_node(
    edges: base_types.GraphType, node: base_types.ComputationNode
) -> FrozenSet[base_types.ComputationEdge]:
    return toolz.pipe(
        edges,
        curried.filter(lambda edge: node == edge.source or node in edge.args),
        frozenset,
    )
