import functools
import inspect
from types import MappingProxyType
from typing import Callable, FrozenSet, Optional, Tuple, Union

import gamla
from gamla.optimized import sync as opt_gamla

from computation_graph import base_types


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
    base_types.edge_args,
    lambda edge: edge.args + (edge.destination,),
    lambda edge: (edge.source, edge.destination),
)

get_all_nodes = gamla.compose_left(gamla.mapcat(get_edge_nodes), frozenset)


edges_to_node_id_map = gamla.compose_left(
    gamla.mapcat(get_edge_nodes), gamla.unique, enumerate, gamla.map(reversed), dict
)

_CallableOrNode = Union[Callable, base_types.ComputationNode]


def _supported_signature(signature: base_types.NodeSignature):
    return not signature.optional_kwargs and not (
        signature.kwargs and signature.is_args
    )


def make_computation_node(func: _CallableOrNode) -> base_types.ComputationNode:
    if isinstance(func, base_types.ComputationNode):
        return func

    return base_types.ComputationNode(
        name=_infer_callable_name(func),
        func=func,
        signature=gamla.pipe(
            func, _infer_callable_signature, gamla.assert_that(_supported_signature)
        ),
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
                gamla.mapcat(lambda edge: (edge.source, *edge.args)),
                frozenset,
                gamla.contains,
            )
        ),
        frozenset,
    )


def infer_graph_sink(edges: base_types.GraphType) -> base_types.ComputationNode:
    assert edges, "Empty graphs have no sink."
    leaves = get_leaves(edges)
    assert len(leaves) == 1, f"Cannot determine sink for {edges}, got: {tuple(leaves)}."
    return gamla.head(leaves)


get_incoming_edges_for_node = gamla.compose_left(
    gamla.groupby(base_types.edge_destination),
    gamla.valmap(frozenset),
    gamla.dict_to_getter_with_default(frozenset()),
)


get_terminals = gamla.compose_left(
    get_all_nodes, gamla.filter(gamla.attrgetter("is_terminal")), tuple
)


remove_future_edges = gamla.compose(tuple, opt_gamla.remove(base_types.edge_is_future))


def make_source():
    return make_source_with_name("unnamed source")


def make_source_with_name(name: str):
    def source():
        raise NotImplementedError(f"pure source [{name}] should never run")

    return make_computation_node(source)
