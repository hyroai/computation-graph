import asyncio
import dataclasses
import itertools
import logging
import typing
from typing import Any, Callable, Dict, FrozenSet, Iterable, Set, Tuple, Type

import gamla
import immutables
import toposort
import typeguard
from gamla.optimized import async_functions as opt_async_gamla
from gamla.optimized import sync as opt_gamla

from computation_graph import base_types, graph, signature
from computation_graph.composers import debug, lift


def _transpose_graph(
    graph: Dict[base_types.ComputationNode, Set[base_types.ComputationNode]]
) -> Dict[base_types.ComputationNode, Set[base_types.ComputationNode]]:
    return opt_gamla.pipe(
        graph, dict.keys, opt_gamla.groupby_many(graph.get), opt_gamla.valmap(set)
    )


_toposort_nodes: Callable[
    [base_types.GraphType], Tuple[FrozenSet[base_types.ComputationNode], ...],
] = opt_gamla.compose_left(
    opt_gamla.groupby_many(base_types.edge_sources),
    opt_gamla.valmap(
        opt_gamla.compose_left(opt_gamla.map(base_types.edge_destination), set)
    ),
    _transpose_graph,
    toposort.toposort,
)


_incoming_edge_options = opt_gamla.compose_left(
    graph.get_incoming_edges_for_node,
    opt_gamla.after(
        opt_gamla.compose_left(
            opt_gamla.groupby(base_types.edge_key),
            opt_gamla.valmap(gamla.sort_by(base_types.edge_priority)),
            dict.values,
            opt_gamla.star(itertools.product),
            opt_gamla.map(tuple),
        )
    ),
)


def _get_args(
    edges_to_results: Dict[base_types.ComputationEdge, Tuple[base_types.Result, ...]]
) -> Tuple[base_types.Result, ...]:
    return opt_gamla.pipe(
        edges_to_results,
        opt_gamla.keyfilter(gamla.attrgetter("args")),
        dict.values,
        gamla.head,
    )


_get_kwargs = opt_gamla.compose_left(
    opt_gamla.keyfilter(
        opt_gamla.compose_left(base_types.edge_key, gamla.not_equals("*args"))
    ),
    opt_gamla.keymap(base_types.edge_key),
    opt_gamla.valmap(gamla.head),
)


_NodeToResults = Dict[base_types.ComputationNode, base_types.Result]
_ComputationInput = Tuple[Tuple[base_types.Result, ...], Dict[str, base_types.Result]]


def _get_computation_input(
    node: base_types.ComputationNode,
    incoming_edges: base_types.GraphType,
    results: Tuple[Tuple[base_types.Result, ...], ...],
) -> _ComputationInput:
    if node.signature.is_kwargs:
        assert (
            len(results) == 1
        ), f"signature for {base_types.pretty_print_function_name(node.func)} contains `**kwargs`. This is considered unary, meaning one incoming edge, but we got more than one: {incoming_edges}."
        return gamla.head(results), {}
    edges_to_results = dict(zip(incoming_edges, results))
    return (
        _get_args(edges_to_results) if node.signature.is_args else (),
        _get_kwargs(edges_to_results),
    )


def _type_check(node: base_types.ComputationNode, result):
    return_typing = typing.get_type_hints(node.func).get("return", None)
    if return_typing:
        try:
            typeguard.check_type(str(node), result, return_typing)
        except TypeError as e:
            logging.error([node.func.__code__, e])


_SingleNodeSideEffect = Callable[[base_types.ComputationNode, Any], None]


def _run_node(is_async: bool, side_effect: _SingleNodeSideEffect) -> Callable:
    if is_async:

        @opt_async_gamla.star
        async def run_node(node, edges_leading_to_node, values):
            args, kwargs = _get_computation_input(node, edges_leading_to_node, values)
            result = node.func(*args, **kwargs)
            result = await gamla.to_awaitable(result)
            side_effect(node, result)
            return result

    else:

        @opt_gamla.star
        def run_node(node, edges_leading_to_node, values):
            args, kwargs = _get_computation_input(node, edges_leading_to_node, values)
            result = node.func(*args, **kwargs)
            side_effect(node, result)
            return result

    return run_node


def _merge_immutable(x, y):
    return x.update(y)


def _process_single_layer(
    f: Callable[[_NodeToResults, base_types.ComputationNode], _NodeToResults]
) -> Callable[[_NodeToResults, FrozenSet[base_types.ComputationNode]], _NodeToResults,]:
    return gamla.compose_left(
        gamla.pack,
        gamla.explode(1),
        gamla.map(f),
        opt_gamla.reduce(_merge_immutable, immutables.Map()),
    )


class _DepNotFoundError(Exception):
    pass


def _edges_to_values(
    node_to_result: Callable[[base_types.ComputationNode], base_types.Result]
) -> Callable[
    [Iterable[base_types.ComputationEdge]], Tuple[Tuple[base_types.Result, ...], ...]
]:
    return opt_gamla.maptuple(
        opt_gamla.compose_left(
            base_types.edge_sources, opt_gamla.maptuple(node_to_result)
        )
    )


def _assoc_immutable(d, k, v):
    return d.set(k, v)


_get_node_input = opt_gamla.compose_left(
    # base_types.ComputationNode, Iterable[Tuple[base_types.ComputationEdge, ...]], edge_to_values.
    gamla.pack,
    gamla.explode(1),
    # Iterable[Tuple[node, incoming_edges, edge_to_values]]
    opt_gamla.map(
        gamla.excepts(
            _DepNotFoundError,
            lambda _: None,
            opt_gamla.juxt(
                gamla.head,
                gamla.second,
                opt_gamla.star(
                    lambda _, edges, edges_to_values: edges_to_values(edges)
                ),
            ),
        )
    ),
    gamla.remove(gamla.equals(None)),
    # Iterable[Tuple[node, edge_option, tuple[tuple[result]]]]
    tuple,
    gamla.translate_exception(gamla.head, StopIteration, _DepNotFoundError),
)


def _populate_reducer_state(
    handled_exceptions, is_async: bool, edges, f: Callable[..., base_types.Result]
) -> Callable[[Callable], Callable]:
    handled_exceptions = (*handled_exceptions, base_types.SkipComputationError)
    incoming_edges_opts = _incoming_edge_options(edges)
    if is_async:

        @opt_async_gamla.star
        async def process_node(
            accumulated_results: _NodeToResults, node: base_types.ComputationNode
        ) -> _NodeToResults:
            try:
                return _assoc_immutable(
                    accumulated_results,
                    node,
                    await f(  # type: ignore
                        node,
                        incoming_edges_opts(node),
                        _edges_to_values(
                            gamla.translate_exception(
                                accumulated_results.__getitem__,
                                KeyError,
                                _DepNotFoundError,
                            )
                        ),
                    ),
                )
            except handled_exceptions:
                return accumulated_results

        return process_node

    else:

        @opt_gamla.star
        def process_node(
            accumulated_results: _NodeToResults, node: base_types.ComputationNode
        ) -> _NodeToResults:
            try:
                return _assoc_immutable(
                    accumulated_results,
                    node,
                    f(
                        node,
                        incoming_edges_opts(node),
                        _edges_to_values(
                            gamla.translate_exception(
                                accumulated_results.__getitem__,
                                KeyError,
                                _DepNotFoundError,
                            )
                        ),
                    ),
                )
            except handled_exceptions:
                return accumulated_results

        return process_node


_is_graph_async = opt_gamla.compose_left(
    opt_gamla.mapcat(lambda edge: (edge.source, *edge.args)),
    opt_gamla.remove(gamla.equals(None)),
    opt_gamla.map(gamla.attrgetter("func")),
    gamla.anymap(asyncio.iscoroutinefunction),
)


_assert_no_unwanted_ambiguity = gamla.compose_left(
    base_types.ambiguity_groups,
    gamla.assert_that_with_message(
        gamla.len_equals(0),
        gamla.wrap_str(
            "There are multiple edges with the same destination, key and priority in the computation graph!: {}"
        ),
    ),
)


def _make_runner(single_node_runner):
    return gamla.reduce_curried(
        _process_single_layer(
            gamla.first(
                single_node_runner, gamla.head, exception_type=_DepNotFoundError
            )
        ),
        immutables.Map(),
    )


def _combine_inputs_with_edges(
    inputs: _NodeToResults,
) -> Callable[[base_types.GraphType], base_types.GraphType]:
    def replace_source(edge):
        assert edge.source, "only supports singular edges for now"

        if edge.source not in inputs:
            return None

        return dataclasses.replace(
            edge,
            is_future=False,
            source=graph.make_computation_node(
                debug.name_callable(lift.always(inputs[edge.source]), edge.source.name)
            ),
        )

    return opt_gamla.compose_left(
        opt_gamla.map(gamla.when(base_types.edge_is_future, replace_source)),
        opt_gamla.remove(gamla.equals(None)),
        tuple,
    )


def _to_callable_with_side_effect_for_single_and_multiple(
    single_node_side_effect: _SingleNodeSideEffect,
    all_nodes_side_effect: Callable,
    edges: base_types.GraphType,
    handled_exceptions: FrozenSet[Type[Exception]],
) -> Callable[[_NodeToResults, _NodeToResults], _NodeToResults]:
    edges = gamla.pipe(
        edges,
        gamla.unique,
        tuple,
        gamla.side_effect(_assert_composition_is_valid),
        gamla.side_effect(_assert_no_unwanted_ambiguity),
    )
    is_async = _is_graph_async(edges)

    def runner(edges):
        return _make_runner(
            _populate_reducer_state(
                handled_exceptions,
                is_async,
                edges,
                gamla.compose(
                    _run_node(is_async, single_node_side_effect), _get_node_input
                ),
            )
        )

    if is_async:

        async def final_runner(edges):
            return await gamla.pipe(
                _toposort_nodes(edges),
                runner(edges),
                dict,
                gamla.side_effect(all_nodes_side_effect(edges)),
            )

    else:

        def final_runner(edges):
            return gamla.pipe(
                _toposort_nodes(edges),
                runner(edges),
                dict,
                gamla.side_effect(all_nodes_side_effect(edges)),
            )

    return (_async_graph_reducer if is_async else _graph_reducer)(
        gamla.compose(
            final_runner, lambda inputs: _combine_inputs_with_edges(inputs)(edges)
        )
    )


def _graph_reducer(graph_callable):
    def reducer(prev: _NodeToResults, sources: _NodeToResults) -> _NodeToResults:
        return {**prev, **(graph_callable({**prev, **sources}))}

    return reducer


def _async_graph_reducer(graph_callable):
    async def reducer(prev: _NodeToResults, sources: _NodeToResults) -> _NodeToResults:
        return {**prev, **(await graph_callable({**prev, **sources}))}

    return reducer


to_callable_with_side_effect = gamla.curry(
    _to_callable_with_side_effect_for_single_and_multiple
)(_type_check)

# Use the second line if you want to see the winning path in the computation graph (a little slower).
to_callable = to_callable_with_side_effect(gamla.just(gamla.just(None)))
# to_callable = to_callable_with_side_effect(graphviz.computation_trace('utterance_computation.dot'))


def _node_is_properly_composed(
    node_to_incoming_edges: base_types.GraphType,
) -> Callable[[base_types.ComputationNode], bool]:
    return gamla.compose_left(
        graph.unbound_signature(node_to_incoming_edges),
        signature.parameters,
        gamla.len_equals(0),
    )


def _assert_composition_is_valid(g):
    return opt_gamla.pipe(
        g,
        graph.get_all_nodes,
        opt_gamla.remove(
            _node_is_properly_composed(graph.get_incoming_edges_for_node(g))
        ),
        opt_gamla.map(gamla.wrap_str("{0} at {0.func.__code__}")),
        tuple,
        gamla.assert_that_with_message(
            gamla.wrap_str("Bad composition for: {}"), gamla.len_equals(0)
        ),
    )


def to_callable_strict(
    g: base_types.GraphType,
) -> Callable[[_NodeToResults, _NodeToResults], _NodeToResults,]:
    return gamla.compose(
        gamla.star(to_callable(g, frozenset())), gamla.map(immutables.Map), gamla.pack
    )
