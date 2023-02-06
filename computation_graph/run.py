import asyncio
import dataclasses
import itertools
import logging
import time
import typing
from typing import Any, Callable, Dict, FrozenSet, Iterable, Optional, Set, Tuple, Type

import gamla
import immutables
import termcolor
import toposort
import typeguard
from gamla.optimized import async_functions as opt_async_gamla
from gamla.optimized import sync as opt_gamla

from computation_graph import base_types, graph, signature


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

_get_args: Callable[
    [Dict[base_types.ComputationEdge, Tuple[base_types.Result, ...]]],
    Tuple[base_types.Result, ...],
] = gamla.compose_left(
    opt_gamla.keyfilter(base_types.edge_args), dict.values, gamla.head
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


def _profile(node, time_started: float):
    elapsed = time.perf_counter() - time_started
    if elapsed <= 0.1:
        return
    logging.warning(
        termcolor.colored(
            f"function took {elapsed:.2f} seconds: {base_types.pretty_print_function_name(node.func)}",
            color="red",
        )
    )


def _run_node(
    is_async: bool, side_effect: _SingleNodeSideEffect
) -> Callable[..., base_types.Result]:
    if is_async:

        @opt_async_gamla.star
        async def run_node(node, edges_leading_to_node, values) -> base_types.Result:
            args, kwargs = _get_computation_input(node, edges_leading_to_node, values)
            before = time.perf_counter()
            result = await gamla.to_awaitable(node.func(*args, **kwargs))
            side_effect(node, result)
            _profile(node, before)
            return result

    else:

        @opt_gamla.star
        def run_node(node, edges_leading_to_node, values) -> base_types.Result:
            args, kwargs = _get_computation_input(node, edges_leading_to_node, values)
            before = time.perf_counter()
            result = node.func(*args, **kwargs)
            side_effect(node, result)
            _profile(node, before)
            return result

    return run_node


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
    # Iterable[Tuple[node, edge_option, Tuple[Tuple[base_types.Result, ...], ...], ...]]
    tuple,
    gamla.translate_exception(gamla.head, StopIteration, _DepNotFoundError),
)


def _make_process_node(
    handled_exceptions: Tuple[Type[Exception], ...],
    is_async: bool,
    incoming_edges_options: Callable[
        [base_types.ComputationNode], Tuple[base_types.GraphType, ...]
    ],
    run_node: Callable[..., base_types.Result],
) -> Callable[
    [_NodeToResults, base_types.ComputationNode],
    Optional[Tuple[base_types.ComputationNode, base_types.Result]],
]:
    handled_exceptions = (*handled_exceptions, base_types.SkipComputationError)
    if is_async:

        @opt_async_gamla.star
        async def process_node(
            accumulated_results: _NodeToResults, node: base_types.ComputationNode
        ) -> Optional[Tuple[base_types.ComputationNode, base_types.Result]]:
            try:
                return (
                    node,
                    await run_node(  # type: ignore
                        node,
                        incoming_edges_options(node),
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
                return None

    else:

        @opt_gamla.star
        def process_node(
            accumulated_results: _NodeToResults, node: base_types.ComputationNode
        ) -> Optional[Tuple[base_types.ComputationNode, base_types.Result]]:
            try:
                return (
                    node,
                    run_node(
                        node,
                        incoming_edges_options(node),
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
                return None

    return process_node


_is_graph_async = opt_gamla.compose_left(
    opt_gamla.mapcat(lambda edge: (edge.source, *edge.args)),
    opt_gamla.remove(gamla.equals(None)),
    opt_gamla.map(base_types.node_implementation),
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


def _future_edge_to_regular_edge_with_placeholder(
    source_to_placeholder: dict[base_types.ComputationNode, base_types.ComputationNode]
) -> Callable[[base_types.ComputationEdge], base_types.ComputationEdge]:
    def replace_source(edge):
        assert edge.source, "only supports singular edges for now"

        return dataclasses.replace(
            edge, is_future=False, source=source_to_placeholder[edge.source]
        )

    return replace_source


_map_future_source_to_placeholder = opt_gamla.compose_left(
    opt_gamla.map(
        opt_gamla.pair_right(
            opt_gamla.compose_left(
                gamla.attrgetter("name"), graph.make_source_with_name
            )
        )
    ),
    gamla.frozendict,
)
_graph_to_future_sources = opt_gamla.compose_left(
    opt_gamla.filter(base_types.edge_is_future),
    # We assume that future edges cannot be with multiple sources
    opt_gamla.map(base_types.edge_source),
    frozenset,
)


def _to_callable_with_side_effect_for_single_and_multiple(
    single_node_side_effect: _SingleNodeSideEffect,
    all_nodes_side_effect: Callable,
    edges: base_types.GraphType,
    handled_exceptions: Tuple[Type[Exception], ...],
) -> Callable[[_NodeToResults, _NodeToResults], _NodeToResults]:
    future_sources = _graph_to_future_sources(edges)
    future_source_to_placeholder = _map_future_source_to_placeholder(future_sources)
    edges = gamla.pipe(
        edges,
        gamla.unique,
        opt_gamla.map(
            opt_gamla.when(
                base_types.edge_is_future,
                _future_edge_to_regular_edge_with_placeholder(
                    future_source_to_placeholder
                ),
            )
        ),
        tuple,
        gamla.side_effect(_assert_composition_is_valid),
        gamla.side_effect(_assert_no_unwanted_ambiguity),
    )
    is_async = _is_graph_async(edges)
    placeholder_to_future_source = opt_gamla.pipe(
        future_source_to_placeholder, gamla.itemmap(lambda k_v: (k_v[1], k_v[0]))
    )
    topological_layers = opt_gamla.pipe(
        edges,
        _toposort_nodes,
        gamla.map_filter_empty(
            gamla.compose_left(
                gamla.remove(gamla.contains(placeholder_to_future_source)), frozenset
            )
        ),
        tuple,
    )
    translate_source_to_placeholder = opt_gamla.compose_left(
        opt_gamla.keyfilter(gamla.contains(future_sources)),
        opt_gamla.keymap(future_source_to_placeholder.__getitem__),
    )
    all_node_side_effects_on_edges = all_nodes_side_effect(edges)
    reduce_layers = _make_reduce_layers(
        edges, handled_exceptions, is_async, single_node_side_effect
    )

    if is_async:

        async def final_runner(sources_to_values):
            return await gamla.pipe(
                topological_layers,
                reduce_layers(
                    immutables.Map(translate_source_to_placeholder(sources_to_values))
                ),
                dict,
                gamla.side_effect(all_node_side_effects_on_edges),
            )

    else:

        def final_runner(sources_to_values):
            return gamla.pipe(
                topological_layers,
                reduce_layers(
                    immutables.Map(translate_source_to_placeholder(sources_to_values))
                ),
                dict,
                gamla.side_effect(all_node_side_effects_on_edges),
            )

    return (_async_graph_reducer if is_async else _graph_reducer)(final_runner)


def _make_reduce_layers(
    edges: base_types.GraphType,
    handled_exceptions: Tuple[Type[Exception], ...],
    is_async: bool,
    single_node_side_effect: _SingleNodeSideEffect,
) -> Callable[[immutables.Map], Callable[[Tuple[FrozenSet, ...]], immutables.Map]]:
    process_node_over_input_options = _make_process_node(
        handled_exceptions,
        is_async,
        _incoming_edge_options(edges),
        gamla.compose(_run_node(is_async, single_node_side_effect), _get_node_input),
    )
    reduce_single_layer = gamla.compose_left(
        gamla.pack,
        gamla.juxt(
            gamla.head,
            gamla.compose_left(
                gamla.explode(1),
                gamla.map_filter_empty(
                    gamla.first(
                        process_node_over_input_options,
                        gamla.just(None),
                        exception_type=_DepNotFoundError,
                    )
                ),
                tuple,
            ),
        ),
        opt_gamla.star(
            lambda prev_results, current_results: prev_results.update(current_results)
        ),
    )
    return gamla.curry(gamla.reduce_curried)(reduce_single_layer)


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


def _assert_composition_is_valid(g: base_types.GraphType):
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
