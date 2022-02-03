from typing import Callable, FrozenSet

import gamla
from gamla.optimized import sync as opt_gamla

from computation_graph import base_types, signature

remove_future_edges = gamla.compose(tuple, opt_gamla.remove(base_types.edge_is_future))


get_edge_nodes = gamla.ternary(
    base_types.edge_args,
    lambda edge: edge.args + (edge.destination,),
    lambda edge: (edge.source, edge.destination),
)

get_all_nodes = gamla.compose_left(gamla.mapcat(get_edge_nodes), frozenset)


edges_to_node_id_map = gamla.compose_left(
    gamla.mapcat(get_edge_nodes), gamla.unique, enumerate, gamla.map(reversed), dict
)


def make_computation_node(
    func: base_types.CallableOrNode,
) -> base_types.ComputationNode:
    if isinstance(func, base_types.ComputationNode):
        return func

    return base_types.ComputationNode(
        name=signature.name(func),
        func=func,
        signature=gamla.pipe(
            func,
            signature.from_callable,
            gamla.assert_that_with_message(
                gamla.just(str(func)), signature.is_supported
            ),
        ),
        is_terminal=False,
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


sink_excluding_terminals = gamla.compose_left(
    get_leaves,
    gamla.remove(gamla.attrgetter("is_terminal")),
    tuple,
    gamla.assert_that(gamla.len_equals(1)),
    gamla.head,
)

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

    source.__name__ = f"source:{name}"
    return make_computation_node(source)


@gamla.curry
def make_terminal(name: str, func: Callable) -> base_types.ComputationNode:
    return base_types.ComputationNode(
        name=name, func=func, signature=signature.from_callable(func), is_terminal=True
    )


_keep_not_in_bound_kwargs = gamla.compose_left(
    opt_gamla.map(base_types.edge_key),
    gamla.filter(gamla.identity),
    frozenset,
    gamla.contains,
    gamla.remove,
)


@gamla.curry
def unbound_signature(
    node_to_incoming_edges, node: base_types.ComputationNode
) -> base_types.NodeSignature:
    """Computes the new signature of unbound variables after considering internal edges."""
    incoming_edges = node_to_incoming_edges(node)
    keep_not_in_bound_kwargs = _keep_not_in_bound_kwargs(incoming_edges)
    return base_types.NodeSignature(
        is_kwargs=False,
        is_args=node.signature.is_args
        and not any(edge.args for edge in incoming_edges),
        kwargs=tuple(keep_not_in_bound_kwargs(node.signature.kwargs)),
        optional_kwargs=tuple(keep_not_in_bound_kwargs(node.signature.optional_kwargs)),
    )
