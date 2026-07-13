import dataclasses
import functools
from typing import Callable, Dict, FrozenSet, Optional, Tuple

import gamla
from gamla.optimized import sync as opt_gamla

from computation_graph import base_types
from computation_graph import infer_sink as _infer_sink_module
from computation_graph import signature

get_edge_nodes = opt_gamla.ternary(
    base_types.edge_args,
    lambda edge: edge.args + (edge.destination,),
    lambda edge: (edge.source, edge.destination),
)


def get_all_nodes_with_filter(
    filter_nodes: Optional[Callable[[base_types.ComputationNode], bool]]
):
    f = opt_gamla.mapcat(get_edge_nodes)

    if filter_nodes is not None:
        f = opt_gamla.compose_left(f, frozenset, opt_gamla.filter(filter_nodes))

    return functools.cache(opt_gamla.compose_left(f, frozenset))


get_all_nodes = get_all_nodes_with_filter(None)


# TODO(nitzo): Rethink the API here.
def get_all_nodes_from_graph_with_filter(
    filter_nodes: Optional[Callable[[base_types.ComputationNode], bool]]
) -> Callable[[base_types.GraphType], FrozenSet[base_types.ComputationNode]]:
    return opt_gamla.compose_left(
        gamla.attrgetter("edges"), get_all_nodes_with_filter(filter_nodes)
    )


edges_to_node_id_map = opt_gamla.compose_left(
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


get_incoming_edges_for_node = opt_gamla.compose_left(
    opt_gamla.groupby(base_types.edge_destination),
    opt_gamla.valmap(frozenset),
    gamla.dict_to_getter_with_default(frozenset()),
)


get_terminals = opt_gamla.compose_left(
    get_all_nodes, opt_gamla.filter(base_types.node_is_terminal), tuple
)


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


_keep_not_in_bound_kwargs = opt_gamla.compose_left(
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
        is_kwargs=node.signature.is_kwargs
        and "**kwargs" not in tuple(opt_gamla.map(base_types.edge_key)(incoming_edges))
        and signature.is_unary(node.signature),
        is_args=node.signature.is_args
        and not any(edge.args for edge in incoming_edges),
        kwargs=tuple(keep_not_in_bound_kwargs(node.signature.kwargs)),
        optional_kwargs=tuple(keep_not_in_bound_kwargs(node.signature.optional_kwargs)),
    )


def _node_in_edge_args(
    x: base_types.CallableOrNode,
) -> Callable[[base_types.ComputationEdge], bool]:
    node = make_computation_node(x)

    def node_in_edge_args(edge: base_types.ComputationEdge) -> bool:
        return node in edge.args

    return node_in_edge_args


def _replace_source_in_edges(
    original: base_types.CallableOrNode, replacement: base_types.CallableOrNode
) -> Callable[[base_types.GraphType], base_types.GraphType]:
    return gamla.compose(
        gamla.star(base_types.GraphType),
        gamla.stack(
            [
                gamla.compose(
                    transform_edges(
                        edge_source_equals(original), replace_edge_source(replacement)
                    ),
                    transform_edges(
                        _node_in_edge_args(original),
                        _replace_edge_source_args(original, replacement),
                    ),
                ),
                gamla.identity,
            ]
        ),
        tuple,
        gamla.juxt(gamla.attrgetter("edges"), gamla.attrgetter("sink")),
    )


traverse_forward: Callable[
    [frozenset[base_types.ComputationEdge]],
    Callable[[base_types.CallableOrNode], Tuple[base_types.ComputationNode, ...]],
] = opt_gamla.compose_left(
    gamla.mapcat(
        opt_gamla.compose_left(
            gamla.juxt(base_types.edge_sources, base_types.edge_destination),
            gamla.explode(0),
        )
    ),
    gamla.groupby_many_reduce(
        opt_gamla.compose_left(gamla.head, gamla.wrap_tuple),
        lambda destinations, e: {*(destinations if destinations else ()), e[1]},
    ),
    gamla.dict_to_getter_with_default(()),
    gamla.before(make_computation_node),
)


@gamla.curry
def replace_source(
    original: base_types.CallableOrNode,
    replacement: base_types.CallableOrNodeOrGraph,
    current_graph: base_types.GraphType,
) -> base_types.GraphType:
    if make_computation_node(original) not in get_all_nodes(current_graph.edges):
        return current_graph

    if gamla.is_instance(base_types.CallableOrNode)(replacement):
        return _replace_source_in_edges(original, replacement)(current_graph)  # type: ignore

    if isinstance(replacement, base_types.GraphType):
        return gamla.pipe(
            current_graph,
            _replace_source_in_edges(original, replacement.sink),
            lambda transformed_graph: base_types.GraphType(
                base_types.merge_edges(transformed_graph.edges, replacement.edges),
                transformed_graph.sink,
            ),
        )

    raise RuntimeError(f"Unsupported relacement graph {replacement}")


def replace_destination(
    original: base_types.CallableOrNode, replacement: base_types.CallableOrNode
) -> Callable[[base_types.GraphType], base_types.GraphType]:
    original = make_computation_node(original)
    replacement = make_computation_node(replacement)
    return gamla.compose(
        gamla.star(base_types.GraphType),
        gamla.stack(
            [
                transform_edges(
                    edge_destination_equals(original),
                    _replace_edge_destination(replacement),
                ),
                gamla.when(gamla.equals(original), gamla.just(replacement)),
            ]
        ),
        tuple,
        gamla.juxt(gamla.attrgetter("edges"), gamla.attrgetter("sink")),
    )


def replace_node(
    original: base_types.CallableOrNode, replacement: base_types.CallableOrNode
) -> Callable[[base_types.GraphType], base_types.GraphType]:
    return gamla.compose(
        replace_source(original, replacement),
        replace_destination(original, replacement),
    )


def replace_nodes(
    node_map: Dict[base_types.ComputationNode, base_types.ComputationNode]
) -> Callable[[base_types.GraphType], base_types.GraphType]:
    """Replace many nodes at once (sources, destinations and sink) in a single pass."""

    def replace_nodes(g: base_types.GraphType) -> base_types.GraphType:
        if not node_map:
            return g
        new_edges = []
        for edge in g.edges:
            destination = node_map.get(edge.destination, edge.destination)
            if edge.args:
                args = tuple(node_map.get(arg, arg) for arg in edge.args)
                if destination is edge.destination and all(
                    new is old for new, old in zip(args, edge.args)
                ):
                    new_edges.append(edge)
                else:
                    new_edges.append(
                        dataclasses.replace(edge, args=args, destination=destination)
                    )
            else:
                assert edge.source is not None
                source = node_map.get(edge.source, edge.source)
                if source is edge.source and destination is edge.destination:
                    new_edges.append(edge)
                else:
                    new_edges.append(
                        dataclasses.replace(
                            edge, source=source, destination=destination
                        )
                    )
        return base_types.GraphType(
            edges=frozenset(new_edges), sink=node_map.get(g.sink, g.sink)
        )

    return replace_nodes


def transform_edges(
    query: Callable[[base_types.ComputationEdge], bool],
    edge_mapper: Callable[[base_types.ComputationEdge], base_types.ComputationEdge],
):
    return _operate_on_edges(
        _split_by_condition(query),
        opt_gamla.compose_left(gamla.map(edge_mapper), tuple),
    )


def edge_source_equals(
    x: base_types.CallableOrNode,
) -> Callable[[base_types.ComputationEdge], bool]:
    x = make_computation_node(x)

    def edge_source_equals(edge):
        return edge.source == x

    return edge_source_equals


def edge_destination_equals(
    x: base_types.CallableOrNode,
) -> Callable[[base_types.ComputationEdge], bool]:
    x = make_computation_node(x)

    def edge_destination_equals(edge):
        return edge.destination == x

    return edge_destination_equals


def replace_edge_source(
    replacement: base_types.CallableOrNode,
) -> Callable[[base_types.ComputationEdge], base_types.ComputationEdge]:
    return gamla.dataclass_replace("source", make_computation_node(replacement))


def _replace_edge_source_args(
    original: base_types.CallableOrNode, replacement: base_types.CallableOrNode
):
    def replace_edge_source_args(
        edge: base_types.ComputationEdge,
    ) -> base_types.ComputationEdge:
        return dataclasses.replace(
            edge,
            args=gamla.pipe(
                edge.args,
                gamla.map(
                    gamla.when(
                        gamla.equals(make_computation_node(original)),
                        gamla.just(make_computation_node(replacement)),
                    )
                ),
                tuple,
            ),
        )

    return replace_edge_source_args


def _replace_edge_destination(
    replacement: base_types.CallableOrNode,
) -> Callable[[base_types.ComputationEdge], base_types.ComputationEdge]:
    return gamla.dataclass_replace("destination", make_computation_node(replacement))


def _operate_on_edges(selector, transformation):
    return gamla.compose(
        gamla.star(
            lambda match, rest: base_types.merge_edges(rest, transformation(match))
        ),
        selector,
    )


def _split_by_condition(condition):
    return opt_gamla.compose_left(
        gamla.bifurcate(gamla.filter(condition), gamla.remove(condition)),
        gamla.map(tuple),
        tuple,
    )


def merge_graphs(
    *graphs, sink_node_or_graph: base_types.ComputationNode | base_types.GraphType
) -> base_types.GraphType:
    assert all(
        isinstance(x, base_types.GraphType) for x in graphs
    ), f"{graphs} not a graph"
    assert (
        not base_types.is_computation_graph(sink_node_or_graph)
        or sink_node_or_graph in graphs
    ), "sink graph must be one of the graphs being merged"

    new_g = gamla.pipe(
        graphs,
        gamla.remove(gamla.equals(base_types.EMPTY_GRAPH)),
        tuple,
        gamla.pair_right(
            gamla.just(
                sink_node_or_graph.sink
                if isinstance(sink_node_or_graph, base_types.GraphType)
                else sink_node_or_graph
            )
        ),
        gamla.packstack(
            gamla.compose_left(
                gamla.map(gamla.attrgetter("edges")), gamla.star(base_types.merge_edges)
            ),
            gamla.identity,
        ),
        gamla.star(base_types.GraphType),
    )

    if base_types.is_debug_mode():
        assert new_g.sink == _infer_sink_module.infer_sink(
            new_g.edges
        ), f"Infer sink (new) mismatch: graph.sink: {new_g.sink}, infer_sink: {_infer_sink_module.infer_sink(new_g.edges)}"
    # assert new_g.sink == _infer_sink_module.infer_sink_old(new_g.edges),  f"Infer sink (old) mismatch: graph.sink: {new_g.sink}, infer_sink: {_infer_sink_module.infer_sink_old(new_g.edges)}"

    return base_types.assert_no_unwanted_ambiguity_when_debug_set(new_g)
