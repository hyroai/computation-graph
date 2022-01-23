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

from computation_graph import base_types, graph


def _transpose_graph(
    graph: Dict[base_types.ComputationNode, Set[base_types.ComputationNode]]
) -> Dict[base_types.ComputationNode, Set[base_types.ComputationNode]]:
    return opt_gamla.pipe(
        graph, dict.keys, opt_gamla.groupby_many(graph.get), opt_gamla.valmap(set)
    )


def _get_edge_destination_for_toposort(
    edge: base_types.ComputationEdge,
) -> Tuple[base_types.ComputationNode, bool]:
    return edge.destination, edge.is_future


_toposort_nodes: Callable[
    [base_types.GraphType],
    Tuple[FrozenSet[Tuple[base_types.ComputationNode, bool]], ...],
] = opt_gamla.compose_left(
    opt_gamla.groupby_many(
        gamla.compose_left(
            gamla.juxt(base_types.edge_sources, base_types.edge_is_future),
            gamla.explode(0),
        )
    ),
    opt_gamla.valmap(
        opt_gamla.compose_left(opt_gamla.map(_get_edge_destination_for_toposort), set)
    ),
    _transpose_graph,
    toposort.toposort,
    opt_gamla.map(frozenset),
    tuple,
)


_incoming_edge_options = opt_gamla.compose_left(
    graph.get_incoming_edges_for_node,
    gamla.after(
        opt_gamla.compose_left(
            opt_gamla.groupby(base_types.edge_key),
            opt_gamla.valmap(gamla.sort_by(gamla.attrgetter("priority"))),
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


_get_inner_kwargs = opt_gamla.compose_left(
    opt_gamla.keyfilter(base_types.edge_key),
    opt_gamla.keymap(base_types.edge_key),
    opt_gamla.valmap(gamla.head),
)


_IntermediaryResults = Dict[base_types.ComputationNode, base_types.Result]


NodeToResults = Callable[[base_types.ComputationNode], base_types.Result]


def _get_computation_input(
    node: base_types.ComputationNode,
    incoming_edges: base_types.GraphType,
    results: Tuple[Tuple[base_types.Result, ...], ...],
) -> base_types.ComputationInput:
    if node.signature.is_kwargs:
        assert (
            len(results) == 1
        ), f"signature for {base_types.pretty_print_function_name(node.func)} contains `**kwargs`. This is considered unary, meaning one incoming edge, but we got more than one: {incoming_edges}."
        return base_types.ComputationInput(args=gamla.head(results), kwargs={})
    if (
        not node.signature.is_args
        and sum(
            map(
                opt_gamla.compose_left(base_types.edge_key, gamla.equals(None)),
                incoming_edges,
            )
        )
        == 1
    ):
        return base_types.ComputationInput(
            args=(), kwargs={node.signature.kwargs[0]: gamla.head(gamla.head(results))}
        )
    edges_to_results = dict(zip(incoming_edges, results))
    return base_types.ComputationInput(
        args=_get_args(edges_to_results) if node.signature.is_args else (),
        kwargs=_get_inner_kwargs(edges_to_results),
    )


def _type_check(node: base_types.ComputationNode, result):
    try:
        return_typing = typing.get_type_hints(node.func).get("return", None)
    except TypeError:
        # Does not support `functools.partial`.
        return
    if return_typing:
        try:
            typeguard.check_type(str(node), result, return_typing)
        except TypeError as e:
            logging.error([node.func.__code__, e])


def _apply(f: Callable, node_input: base_types.ComputationInput) -> base_types.Result:
    return f(*node_input.args, **node_input.kwargs)


_SingleNodeSideEffect = Callable[[base_types.ComputationNode, Any], None]


def _run_keeping_choices(
    is_async: bool, side_effect: _SingleNodeSideEffect
) -> Callable:
    if is_async:

        @opt_async_gamla.star
        async def run_keeping_choices(node, edges_leading_to_node, values):
            result = _apply(
                node.func, _get_computation_input(node, edges_leading_to_node, values)
            )
            result = await gamla.to_awaitable(result)
            side_effect(node, result)
            return result

    else:

        @opt_gamla.star
        def run_keeping_choices(node, edges_leading_to_node, values):
            result = _apply(
                node.func, _get_computation_input(node, edges_leading_to_node, values)
            )
            side_effect(node, result)
            return result

    return run_keeping_choices


def _merge_immutable(x, y):
    return x.update(y)


def _handle_node_not_processing(exception, inputs):
    del exception, inputs
    return immutables.Map()


def _process_layer_in_parallel(
    f: Callable[
        [_IntermediaryResults, base_types.ComputationNode], _IntermediaryResults
    ]
) -> Callable[
    [_IntermediaryResults, FrozenSet[base_types.ComputationNode]], _IntermediaryResults,
]:
    return gamla.compose_left(
        gamla.pack,
        gamla.explode(1),
        opt_gamla.map(opt_gamla.star(lambda x, y: (x, y[0]))),
        gamla.map(
            gamla.try_and_excepts(_DepNotFoundError, _handle_node_not_processing, f)
        ),
        opt_gamla.reduce(_merge_immutable, immutables.Map()),
    )


def _dag_layer_reduce(is_async: bool, edges: base_types.GraphType):
    layers = _toposort_nodes(edges)

    def inner(
        f: Callable[
            [_IntermediaryResults, FrozenSet[base_types.ComputationNode]],
            _IntermediaryResults,
        ]
    ):
        if is_async:

            async def inner():
                return await gamla.pipe(
                    layers, gamla.reduce_curried(f, immutables.Map())
                )

        else:

            def inner():
                return gamla.pipe(layers, gamla.reduce_curried(f, immutables.Map()))

        return inner

    return inner


class _DepNotFoundError(Exception):
    pass


def _edges_to_values(
    accumulated_outputs,
) -> Callable[
    [Iterable[base_types.ComputationEdge]], Tuple[Tuple[base_types.Result, ...], ...]
]:
    return opt_gamla.maptuple(
        opt_gamla.compose_left(
            base_types.edge_sources,
            opt_gamla.maptuple(
                gamla.translate_exception(
                    accumulated_outputs.__getitem__, KeyError, _DepNotFoundError
                )
            ),
        )
    )


def _assoc_immutable(d, k, v):
    return d.set(k, v)


@gamla.curry
def _lift_single_runner_to_run_on_many_options(is_async: bool, f):
    return (opt_async_gamla.compose_left if is_async else opt_gamla.compose_left)(
        # Node, edges_options, edge_to_value.
        gamla.pack,
        gamla.explode(1),
        # iterable of (node, edge_option, edge_to_value)
        opt_gamla.map(
            gamla.excepts(
                _DepNotFoundError,
                lambda _: None,
                opt_gamla.compose_left(
                    gamla.bifurcate(
                        gamla.head,
                        gamla.second,
                        opt_gamla.star(
                            lambda _, edges, edges_to_values: edges_to_values(edges)
                        ),
                    )
                ),
            )
        ),
        gamla.remove(gamla.equals(None)),
        # iterable of (node, edge_option, tuple[tuple[result]])
        tuple,
        gamla.translate_exception(gamla.head, StopIteration, _DepNotFoundError),
        f,
    )


def _process_node(
    handled_exceptions,
    is_async: bool,
    get_edge_options: Callable[[base_types.ComputationNode], Any],
) -> Callable[[Callable], Callable]:
    def process_node(f: Callable):
        if is_async:

            @opt_async_gamla.star
            async def process_node(
                accumulated_results: _IntermediaryResults,
                node: base_types.ComputationNode,
            ) -> _IntermediaryResults:
                try:
                    return _assoc_immutable(
                        accumulated_results,
                        node,
                        await f(
                            node,
                            get_edge_options(node),
                            _edges_to_values(accumulated_results),
                        ),
                    )
                except handled_exceptions:
                    return accumulated_results

            return process_node

        else:

            @opt_gamla.star
            def process_node(
                accumulated_results: _IntermediaryResults,
                node: base_types.ComputationNode,
            ) -> _IntermediaryResults:
                try:
                    return _assoc_immutable(
                        accumulated_results,
                        node,
                        f(
                            node,
                            get_edge_options(node),
                            _edges_to_values(accumulated_results),
                        ),
                    )
                except handled_exceptions:
                    return accumulated_results

            return process_node

    return process_node


_is_graph_async = opt_gamla.compose_left(
    opt_gamla.mapcat(lambda edge: (edge.source, *edge.args)),
    opt_gamla.remove(gamla.equals(None)),
    opt_gamla.map(gamla.attrgetter("func")),
    gamla.anymap(asyncio.iscoroutinefunction),
)


_assert_no_unwanted_ambiguity = gamla.compose_left(
    gamla.groupby(
        gamla.juxt(
            base_types.edge_sources,
            gamla.attrgetter("destination"),
            gamla.attrgetter("priority"),
        )
    ),
    gamla.valmap(
        gamla.assert_that_with_message(
            gamla.len_equals(1),
            gamla.just(
                "There are multiple edges with the same source, destination, and priority in the computation graph!"
            ),
        )
    ),
)


def _make_runner(single_node_runner, is_async, edges, handled_exceptions):
    return opt_gamla.compose(
        _dag_layer_reduce(is_async, edges),
        _process_layer_in_parallel,
        _process_node(
            (*handled_exceptions, base_types.SkipComputationError),
            is_async,
            _incoming_edge_options(edges),
        ),
        _lift_single_runner_to_run_on_many_options(is_async),
    )(single_node_runner)


def _combine_inputs_with_edges(edges, inputs: Dict):
    def replace_source(edge):
        assert edge.source, "only supports singular edges for now"

        if edge.source not in inputs:
            return None

        def source():
            return inputs[edge.source]

        return dataclasses.replace(
            edge, is_future=False, source=graph.make_computation_node(source)
        )

    return opt_gamla.pipe(
        edges,
        gamla.map(gamla.when(base_types.edge_is_future, replace_source)),
        gamla.remove(gamla.equals(None)),
        tuple,
    )


def _to_callable_with_side_effect_for_single_and_multiple(
    single_node_side_effect: _SingleNodeSideEffect,
    all_nodes_side_effect: Callable,
    edges: base_types.GraphType,
    handled_exceptions: FrozenSet[Type[Exception]],
) -> Callable:
    edges = gamla.pipe(
        edges, gamla.unique, tuple, gamla.side_effect(_assert_no_unwanted_ambiguity)
    )
    is_async = _is_graph_async(edges)
    if is_async:

        async def runner(inputs):
            edges_with_inputs = _combine_inputs_with_edges(edges, inputs)
            return await gamla.compose_left(
                _make_runner(
                    _run_keeping_choices(is_async, single_node_side_effect),
                    is_async,
                    edges_with_inputs,
                    handled_exceptions,
                ),
                gamla.side_effect(all_nodes_side_effect(edges_with_inputs)),
            )()

    else:

        def runner(inputs):
            edges_with_inputs = _combine_inputs_with_edges(edges, inputs)
            return gamla.compose_left(
                _make_runner(
                    _run_keeping_choices(is_async, single_node_side_effect),
                    is_async,
                    edges_with_inputs,
                    handled_exceptions,
                ),
                gamla.side_effect(all_nodes_side_effect(edges_with_inputs)),
            )()

    return runner


to_callable_with_side_effect = gamla.curry(
    _to_callable_with_side_effect_for_single_and_multiple
)(_type_check)

# Use the second line if you want to see the winning path in the computation graph (a little slower).
to_callable = to_callable_with_side_effect(gamla.just(gamla.just(None)))
# to_callable = to_callable_with_side_effect(graphviz.computation_trace('utterance_computation.dot'))


def to_callable_strict(
    g: base_types.GraphType,
) -> Callable[
    [Dict[base_types.ComputationNode, base_types.Result]],
    Dict[base_types.ComputationNode, base_types.Result],
]:
    return gamla.compose(to_callable(g, frozenset()), immutables.Map)
