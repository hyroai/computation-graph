import functools
import inspect
from types import MappingProxyType
from typing import Callable, FrozenSet, Optional, Tuple, Union

import gamla
from gamla.optimized import sync as opt_gamla

from computation_graph import base_types


def get_all_nodes(edges: base_types.GraphType) -> FrozenSet[base_types.ComputationNode]:
    return gamla.pipe(edges, gamla.mapcat(get_edge_nodes), gamla.unique, frozenset)


def _is_reducer_type(node: Callable) -> bool:
    return "state" in inspect.signature(node).parameters


def _is_star(parameter) -> bool:
    return "*" + parameter.name == str(parameter)


def _is_double_star(parameter) -> bool:
    return "**" + parameter.name == str(parameter)


def _is_default(parameter):
    return parameter.default != parameter.empty


_parameter_name = gamla.attrgetter("name")


@gamla.before(
    gamla.compose_left(
        inspect.signature,
        gamla.attrgetter("parameters"),
        MappingProxyType.values,
        tuple,
    )
)
def _infer_callable_signature(function_parameters: Tuple) -> base_types.NodeSignature:
    return base_types.NodeSignature(
        is_args=gamla.anymap(_is_star)(function_parameters),
        is_kwargs=gamla.anymap(_is_double_star)(function_parameters),
        kwargs=gamla.pipe(
            function_parameters,
            gamla.remove(gamla.anyjuxt(_is_star, _is_double_star)),
            gamla.map(_parameter_name),
            tuple,
        ),
        optional_kwargs=gamla.pipe(
            function_parameters,
            gamla.remove(gamla.anyjuxt(_is_star, _is_double_star)),
            gamla.filter(_is_default),
            gamla.map(_parameter_name),
            tuple,
        ),
    )


def _infer_callable_name(func: Callable) -> str:
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

_CallableOrNode = Union[Callable, base_types.ComputationNode]


def make_computation_node(func: _CallableOrNode) -> base_types.ComputationNode:
    assert func is not base_types.ComputationNode
    if isinstance(func, base_types.ComputationNode):
        return func

    return base_types.ComputationNode(
        name=_infer_callable_name(func),
        func=func,
        is_stateful=_is_reducer_type(func),
        signature=_infer_callable_signature(func),
        is_terminal=False,
    )


@gamla.curry
def make_edge(
    is_future: bool,
    priority: int,
    source: Union[_CallableOrNode, Tuple[_CallableOrNode, ...]],
    destination: _CallableOrNode,
    key: Optional[str] = None,
) -> base_types.ComputationEdge:
    destination_as_node = make_computation_node(destination)
    if isinstance(source, tuple):
        return base_types.ComputationEdge(
            args=tuple(map(make_computation_node, source)),
            destination=destination_as_node,
            priority=priority,
            source=None,
            key=None,
            is_future=is_future,
        )

    return base_types.ComputationEdge(
        source=make_computation_node(source),
        destination=destination_as_node,
        key=key,
        args=(),
        priority=priority,
        is_future=is_future,
    )


make_standard_edge = make_edge(is_future=False, priority=0)
make_future_edge = make_edge(is_future=True, priority=0)


def get_leaves(edges: base_types.GraphType) -> FrozenSet[base_types.ComputationNode]:
    return gamla.pipe(
        edges,
        get_all_nodes,
        gamla.remove(
            gamla.pipe(
                edges,
                remove_future_edges,
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
        signature=_infer_callable_signature(func),
        is_stateful=False,
        is_terminal=True,
    )


get_terminals = gamla.compose_left(
    get_all_nodes, gamla.filter(gamla.attrgetter("is_terminal")), tuple
)


def _aggregator_for_terminal(*args):
    return tuple(args)


DEFAULT_TERMINAL = make_terminal("DEFAULT_TERMINAL", _aggregator_for_terminal)


def connect_default_terminal(edges: base_types.GraphType) -> base_types.GraphType:
    return edges + (
        make_standard_edge(
            source=(infer_graph_sink(edges),), destination=DEFAULT_TERMINAL
        ),
    )


remove_future_edges = opt_gamla.remove(gamla.attrgetter("is_future"))
