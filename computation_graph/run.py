import asyncio
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
        graph.keys(), opt_gamla.groupby_many(graph.get), opt_gamla.valmap(set)
    )


def _get_edge_sources(edge: base_types.ComputationEdge):
    return edge.args or (edge.source,)


_toposort_nodes: Callable[
    [base_types.GraphType], Tuple[FrozenSet[base_types.ComputationNode], ...]
] = opt_gamla.compose_left(
    graph.remove_future_edges,
    opt_gamla.groupby_many(_get_edge_sources),
    opt_gamla.valmap(
        opt_gamla.compose_left(opt_gamla.map(base_types.edge_destination), set)
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


class _NotCoherent(Exception):
    """This exception signals that for a specific set of incoming
    node edges not all paths agree on the result"""


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
        gamla.map(f),
        opt_gamla.reduce(_merge_immutable, immutables.Map()),
    )


def _dag_layer_reduce(is_async, edges):
    """Directed acyclic graph reduction."""

    layers = _toposort_nodes(edges)

    def inner(
        f: Callable[
            [_IntermediaryResults, FrozenSet[base_types.ComputationNode]],
            _IntermediaryResults,
        ]
    ):
        if is_async:

            async def inner(source_to_value):
                return await gamla.pipe(
                    layers, gamla.reduce_curried(f, source_to_value)
                )

        else:

            def inner(source_to_value):
                return gamla.pipe(layers, gamla.reduce_curried(f, source_to_value))

        return inner

    return inner


def _edges_to_values(
    accumulated_outputs,
) -> Callable[
    [Iterable[base_types.ComputationEdge]], Tuple[Tuple[base_types.Result, ...], ...]
]:
    return opt_gamla.maptuple(
        opt_gamla.compose_left(
            _get_edge_sources, opt_gamla.maptuple(accumulated_outputs.__getitem__)
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
            opt_gamla.compose_left(
                gamla.bifurcate(
                    gamla.head,
                    gamla.second,
                    opt_gamla.star(
                        lambda _, edges, edges_to_values: edges_to_values(edges)
                    ),
                )
            )
        ),
        # iterable of (node, edge_option, tuple[tuple[result]])
        opt_gamla.remove(
            opt_gamla.compose(
                opt_gamla.anymap(gamla.inside(_COULD_NOT_COMPUTE)), gamla.nth(2)
            )
        ),
        tuple,
        gamla.assert_that(gamla.len_equals(1)),
        gamla.head,
        f,
    )


def _process_node(
    is_async: bool, get_edge_options: Callable[[base_types.ComputationNode], Any]
) -> Callable[[Callable], Callable]:
    def process_node(f: Callable):
        if is_async:

            @opt_async_gamla.star
            async def process_node(
                accumulated_results: _IntermediaryResults,
                node: base_types.ComputationNode,
            ) -> _IntermediaryResults:
                return _assoc_immutable(
                    accumulated_results,
                    node,
                    await f(
                        node,
                        get_edge_options(node),
                        _edges_to_values(accumulated_results),
                    ),
                )

            return process_node

        else:

            @opt_gamla.star
            def process_node(
                accumulated_results: _IntermediaryResults,
                node: base_types.ComputationNode,
            ) -> _IntermediaryResults:
                return _assoc_immutable(
                    accumulated_results,
                    node,
                    f(
                        node,
                        get_edge_options(node),
                        _edges_to_values(accumulated_results),
                    ),
                )

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
            _get_edge_sources,
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


class _CouldNotCompute:
    pass


_COULD_NOT_COMPUTE = _CouldNotCompute()


def _make_runner(single_node_runner, is_async, edges, handled_exceptions):
    return opt_gamla.compose(
        _dag_layer_reduce(is_async, edges),
        _process_layer_in_parallel,
        _process_node(is_async, _incoming_edge_options(edges)),
        _lift_single_runner_to_run_on_many_options(is_async),
        gamla.excepts(
            (*handled_exceptions, _NotCoherent, base_types.SkipComputationError),
            gamla.just(_COULD_NOT_COMPUTE),
        ),
    )(single_node_runner)


def _to_callable_with_side_effect_for_single_and_multiple(
    single_node_side_effect: _SingleNodeSideEffect,
    all_nodes_side_effect: Callable,
    edges: base_types.GraphType,
    handled_exceptions: FrozenSet[Type[Exception]],
) -> Callable:
    edges = gamla.pipe(
        edges, gamla.unique, tuple, gamla.side_effect(_assert_no_unwanted_ambiguity)
    )
    return gamla.compose_left(
        _make_runner(
            _run_keeping_choices(_is_graph_async(edges), single_node_side_effect),
            _is_graph_async(edges),
            edges,
            handled_exceptions,
        ),
        gamla.side_effect(all_nodes_side_effect(edges)),
    )


to_callable_with_side_effect = gamla.curry(
    _to_callable_with_side_effect_for_single_and_multiple
)(_type_check)

# Use the second line if you want to see the winning path in the computation graph (a little slower).
to_callable = to_callable_with_side_effect(gamla.just(gamla.just(None)))
# to_callable = to_callable_with_side_effect(graphviz.computation_trace('utterance_computation.dot'))


def to_callable_strict(g):
    return gamla.compose(to_callable(g, frozenset()), immutables.Map)
