import functools
import inspect
from typing import Callable, FrozenSet, Optional, Text, Tuple, Union

import gamla

from computation_graph import base_types


def get_all_nodes(edges: base_types.GraphType) -> FrozenSet[base_types.ComputationNode]:
    return gamla.pipe(edges, gamla.mapcat(get_edge_nodes), gamla.unique, frozenset)


def _is_reducer_type(node: Callable) -> bool:
    return "state" in inspect.signature(node).parameters


def infer_callable_signature(func: Callable) -> base_types.NodeSignature:
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


get_edge_nodes = gamla.ternary(
    gamla.attrgetter("args"),
    lambda edge: edge.args + (edge.destination,),
    lambda edge: (edge.source, edge.destination),
)


edges_to_node_id_map = gamla.compose_left(
    gamla.mapcat(get_edge_nodes), gamla.unique, enumerate, gamla.map(reversed), dict
)


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
        signature=infer_callable_signature(func),
        is_terminal=False,
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
    return gamla.pipe(
        edges,
        get_all_nodes,
        gamla.remove(
            gamla.pipe(
                edges,
                gamla.mapcat(lambda edge: (edge.source, *edge.args)),
                frozenset,
                gamla.contains,
            )
        ),
        frozenset,
    )


def infer_graph_sink(edges: base_types.GraphType) -> base_types.ComputationNode:
    leaves = get_leaves(edges)
    assert len(leaves) == 1, f"computation graph has more than one sink: {leaves}"
    return gamla.head(leaves)


def infer_graph_sink_excluding_terminals(
    edges: base_types.GraphType,
) -> base_types.ComputationNode:
    leaves = gamla.pipe(
        edges, get_leaves, gamla.remove(gamla.attrgetter("is_terminal")), tuple
    )
    assert len(leaves) == 1, f"computation graph has more than one sink: {leaves}"
    return gamla.head(leaves)


get_incoming_edges_for_node = gamla.compose_left(
    gamla.groupby(lambda edge: edge.destination),
    gamla.valmap(frozenset),
    gamla.dict_to_getter_with_default(frozenset()),
)


def make_terminal(name: str, func: Callable):
    return base_types.ComputationNode(
        name=name,
        func=func,
        signature=infer_callable_signature(func),
        is_stateful=False,
        is_terminal=True,
    )


get_terminals = gamla.compose_left(
    get_all_nodes, gamla.filter(gamla.attrgetter("is_terminal")), tuple
)


def _aggregator_for_terminal(*args) -> base_types.GraphType:
    return tuple(args)


DEFAULT_TERMINAL = make_terminal("DEFAULT_TERMINAL", _aggregator_for_terminal)


def connect_default_terminal(edges: base_types.GraphType) -> base_types.GraphType:
    return edges + (make_edge((infer_graph_sink(edges),), DEFAULT_TERMINAL),)
