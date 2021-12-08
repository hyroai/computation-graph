import asyncio
import collections
import dataclasses
import itertools
import logging
import pathlib
import sys
import traceback
import typing
from typing import Any, Callable, Dict, FrozenSet, Iterable, Set, Tuple, Type

import gamla
import immutables
import toposort
import typeguard
from gamla.optimized import async_functions as opt_async_gamla
from gamla.optimized import sync as opt_gamla

from computation_graph import base_types, graph


class _ComputationGraphException(Exception):
    pass


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
    opt_gamla.groupby_many(_get_edge_sources),
    opt_gamla.valmap(
        opt_gamla.compose_left(opt_gamla.map(base_types.edge_destination), set)
    ),
    _transpose_graph,
    toposort.toposort,
    opt_gamla.map(frozenset),
    tuple,
)


def _make_outer_computation_input(*args, **kwargs) -> base_types.ComputationInput:
    if "state" in kwargs:
        return base_types.ComputationInput(
            args=args,
            kwargs=gamla.remove_key("state")(kwargs),
            state=dict(kwargs["state"] or {}),
        )

    return base_types.ComputationInput(args=args, kwargs=kwargs)


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

_get_args_helper = opt_gamla.compose_left(
    opt_gamla.keyfilter(gamla.attrgetter("args")),
    dict.values,
    gamla.head,
    opt_gamla.maptuple(gamla.attrgetter("result")),
)


def _get_args(
    edges_to_results: Dict[
        base_types.ComputationEdge, Tuple[base_types.ComputationResult, ...],
    ],
    unbound_signature: base_types.NodeSignature,
    bound_signature: base_types.NodeSignature,
    unbound_input: base_types.ComputationInput,
) -> Tuple[base_types.ComputationResult, ...]:
    if unbound_signature.is_args:
        return unbound_input.args
    if bound_signature.is_args:
        return _get_args_helper(edges_to_results)
    return ()


def _get_unary_computation_input(
    kwargs: Tuple[str, ...],
    value: base_types.ComputationResult,
    unbound_signature: base_types.NodeSignature,
) -> Dict[str, Any]:
    return opt_gamla.pipe(
        unbound_signature.kwargs,
        opt_gamla.remove(
            opt_gamla.anyjuxt(
                gamla.contains(unbound_signature.optional_kwargs), gamla.equals("state")
            )
        ),
        tuple,
        opt_gamla.check(
            opt_gamla.anyjuxt(gamla.len_equals(1), gamla.len_equals(0)),
            _ComputationGraphException(
                f"got a single input function with more than 1 unbound arguments. cannot bind function. {unbound_signature}"
            ),
        ),
        opt_gamla.ternary(gamla.len_equals(1), gamla.identity, gamla.just(kwargs)),
        gamla.head,
        lambda first_kwarg: {first_kwarg: value.result},
    )


def _get_outer_kwargs(
    unbound_signature: base_types.NodeSignature,
    unbound_input: base_types.ComputationInput,
) -> Dict[str, Any]:
    # Optimized because being called a lot.
    d = {}
    for kwarg in unbound_signature.kwargs:
        if kwarg != "state" and kwarg in unbound_input.kwargs:
            d[kwarg] = unbound_input.kwargs[kwarg]
    return d


_get_inner_kwargs = opt_gamla.compose_left(
    opt_gamla.keyfilter(base_types.edge_key),
    dict.items,
    opt_gamla.groupby(opt_gamla.compose_left(gamla.head, base_types.edge_key)),
    opt_gamla.valmap(
        opt_gamla.compose_left(
            gamla.head, gamla.second, gamla.head, gamla.attrgetter("result")
        )
    ),
)


_DecisionsType = Dict[base_types.ComputationNode, base_types.ComputationResult]
_ResultToDecisionsType = Dict[base_types.ComputationResult, _DecisionsType]
_IntermediaryResults = Dict[base_types.ComputationNode, _ResultToDecisionsType]


class _NotCoherent(Exception):
    """This exception signals that for a specific set of incoming
    node edges not all paths agree on the ComputationResult"""


def _check_equal_and_take_one(x, y):
    if x == y:
        return x
    raise _NotCoherent


NodeToResults = Callable[[base_types.ComputationNode], _ResultToDecisionsType]


def _signature_difference(
    sig_a: base_types.NodeSignature, sig_b: base_types.NodeSignature
) -> base_types.NodeSignature:
    return base_types.NodeSignature(
        is_args=(sig_a.is_args != sig_b.is_args),
        # Difference must save the order of the left signature.
        kwargs=tuple(filter(lambda x: x not in sig_b.kwargs, sig_a.kwargs)),
        optional_kwargs=tuple(
            filter(lambda x: x not in sig_b.optional_kwargs, sig_a.optional_kwargs)
        ),
    )


_get_kwargs_from_edges = opt_gamla.compose_left(
    opt_gamla.map(base_types.edge_key), opt_gamla.remove(gamla.equals(None)), tuple
)


def _get_bound_signature(
    is_args: bool, incoming_edges: base_types.GraphType
) -> base_types.NodeSignature:
    return base_types.NodeSignature(
        is_args=is_args and any(edge.args for edge in incoming_edges),
        kwargs=_get_kwargs_from_edges(incoming_edges),
        optional_kwargs=(),
    )


_ChoiceOfOutputForNode = Tuple[
    Tuple[base_types.ComputationResult, _DecisionsType], base_types.ComputationNode,
]


@gamla.curry
def _get_computation_input(
    unbound_input: Callable[[base_types.ComputationNode], base_types.ComputationInput],
    node: base_types.ComputationNode,
    incoming_edges: base_types.GraphType,
    # For each edge, there are multiple values options, each having its own trace.
    values_for_edges_choice: Iterable[Iterable[_ChoiceOfOutputForNode]],
) -> base_types.ComputationInput:
    incoming_edges_no_future = tuple(graph.remove_future_edges(incoming_edges))
    bound_signature = _get_bound_signature(
        node.signature.is_args, incoming_edges_no_future
    )
    unbound_signature = _signature_difference(node.signature, bound_signature)
    results = gamla.pipe(
        values_for_edges_choice,
        opt_gamla.maptuple(opt_gamla.maptuple(_choice_to_value)),
    )

    future_edges_kwargs = opt_gamla.pipe(
        incoming_edges,
        opt_gamla.filter(gamla.attrgetter("is_future")),
        opt_gamla.map(
            opt_gamla.juxt(
                lambda edge: edge.key or edge.source.signature.kwargs[0],
                opt_gamla.compose_left(
                    gamla.attrgetter("source"), unbound_input, gamla.attrgetter("state")
                ),
            )
        ),
        dict,
    )

    if node.signature.is_kwargs:
        assert (
            len(results) == 1
        ), f"signature for {base_types.pretty_print_function_name(node.func)} contains `**kwargs`. This is considered unary, meaning one incoming edge, but we got more than one: {incoming_edges_no_future}."
        return base_types.ComputationInput(
            args=opt_gamla.pipe(
                results,
                gamla.head,
                gamla.head,
                gamla.attrgetter("result"),
                gamla.wrap_tuple,
            ),
            kwargs={},
            state=None,
        )
    if (
        not (unbound_signature.is_args or bound_signature.is_args)
        and sum(
            map(
                opt_gamla.compose_left(base_types.edge_key, gamla.equals(None)),
                incoming_edges,
            )
        )
        == 1
    ):
        return base_types.ComputationInput(
            args=(),
            kwargs=future_edges_kwargs
            or _get_unary_computation_input(
                node.signature.kwargs,
                gamla.head(gamla.head(results)),
                unbound_signature,
            ),
            state=unbound_input(node).state,
        )
    edges_to_results = dict(zip(incoming_edges_no_future, results))
    return base_types.ComputationInput(
        args=_get_args(
            edges_to_results, unbound_signature, bound_signature, unbound_input(node)
        ),
        kwargs={
            **_get_outer_kwargs(unbound_signature, unbound_input(node)),
            **_get_inner_kwargs(edges_to_results),
            **future_edges_kwargs,
        },
        state=unbound_input(node).state,
    )


def _wrap_in_result_if_needed(node: base_types.ComputationNode, result):
    if isinstance(result, base_types.ComputationResult):
        return result
    if node.is_stateful:
        return base_types.ComputationResult(result, result)
    return base_types.ComputationResult(result, None)


def _inject_state(unbound_input: base_types.ComputationInput):
    def inject_state(node_id: int):
        if unbound_input.state is None:
            return unbound_input
        return dataclasses.replace(
            unbound_input,
            state=unbound_input.state[node_id]
            if node_id in unbound_input.state
            else None,
        )

    return inject_state


_choice_to_value: Callable[
    [_ChoiceOfOutputForNode], base_types.ComputationResult
] = opt_gamla.compose_left(gamla.head, gamla.head)

_decisions_from_value_choices = opt_gamla.compose_left(
    gamla.concat,
    gamla.bifurcate(
        opt_gamla.compose_left(
            opt_gamla.map(opt_gamla.compose_left(gamla.head, gamla.second)),
            opt_gamla.reduce(
                opt_gamla.merge_with_reducer(_check_equal_and_take_one),
                immutables.Map(),
            ),
        ),
        opt_gamla.mapdict(opt_gamla.juxt(gamla.second, _choice_to_value)),
    ),
    opt_gamla.merge,
)


@gamla.curry
def _construct_computation_state(
    edges: base_types.GraphType,
    results: _ResultToDecisionsType,
    sink_node: base_types.ComputationNode,
) -> Dict:
    first_result = gamla.head(results)
    return {
        **{sink_node: first_result.state},
        **opt_gamla.pipe(
            results,
            gamla.itemgetter(first_result),
            opt_gamla.juxt(
                opt_gamla.valmap(gamla.attrgetter("state")),
                gamla.compose_left(
                    opt_gamla.valmap(gamla.attrgetter("result")),
                    opt_gamla.keyfilter(
                        gamla.contains(
                            opt_gamla.pipe(
                                edges,
                                opt_gamla.filter(gamla.attrgetter("is_future")),
                                gamla.map(gamla.attrgetter("source")),
                                frozenset,
                            )
                        )
                    ),
                ),
            ),
            opt_gamla.merge,
        ),
    }


def _merge_with_previous_state(
    previous_state: Dict, result: base_types.ComputationResult, state: Dict
) -> base_types.ComputationResult:
    return base_types.ComputationResult(
        result=result,
        # Convert to tuples (node id, state) so this would be hashable.
        state=tuple({**(previous_state or {}), **state}.items()),
    )


def _get_results_from_terminals(
    result_to_dependencies: Callable,
) -> Callable[[Iterable[base_types.ComputationNode]], _DecisionsType]:
    return opt_gamla.compose_left(
        opt_gamla.map(
            opt_gamla.pair_right(
                opt_gamla.compose_left(
                    result_to_dependencies,
                    opt_gamla.ternary(
                        gamla.identity,
                        opt_gamla.compose_left(
                            dict.items,
                            gamla.head,
                            gamla.head,  # Take results, not dependencies
                            gamla.attrgetter("result"),
                        ),
                        gamla.just({}),
                    ),
                )
            )
        ),
        dict,
        opt_gamla.valfilter(gamla.identity),  # Only return terminals with results
    )


def _get_computation_state_from_terminals(
    result_to_dependencies: Callable,
    edges_to_node_id: Callable,
    edges: base_types.GraphType,
) -> Callable[[Iterable[base_types.ComputationNode]], Dict]:
    return opt_gamla.compose_left(
        opt_gamla.map(
            opt_gamla.compose_left(
                opt_gamla.pair_left(result_to_dependencies),
                opt_gamla.ternary(
                    opt_gamla.compose_left(gamla.head, gamla.nonempty),
                    opt_gamla.compose_left(
                        opt_gamla.star(_construct_computation_state(edges)),
                        opt_gamla.keymap(edges_to_node_id),
                    ),
                    gamla.just({}),
                ),
            )
        ),
        opt_gamla.merge,
    )


def _construct_computation_result(edges: base_types.GraphType, edges_to_node_id):
    def construct_computation_result(
        result_to_dependencies: Callable[
            [base_types.ComputationNode], _ResultToDecisionsType
        ]
    ):
        return opt_gamla.pipe(
            edges,
            graph.get_leaves,
            opt_gamla.filter(gamla.attrgetter("is_terminal")),
            frozenset,
            opt_gamla.juxt(
                _get_results_from_terminals(result_to_dependencies),
                _get_computation_state_from_terminals(
                    result_to_dependencies, edges_to_node_id, edges
                ),
            ),
        )

    return construct_computation_result


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


def _apply(node: base_types.ComputationNode, node_input: base_types.ComputationInput):
    return node.func(
        *node_input.args,
        **gamla.add_key_value("state", node_input.state)(node_input.kwargs)
        if node.is_stateful
        else node_input.kwargs,
    )


_SingleNodeSideEffect = Callable[[base_types.ComputationNode, Any], None]


def _run_keeping_choices(
    is_async: bool, side_effect: _SingleNodeSideEffect
) -> Callable:
    def run_keeping_choices(node_to_external_input):
        input_maker = opt_gamla.star(_get_computation_input(node_to_external_input))

        if is_async:

            async def run_keeping_choices(params):
                result = _apply(params[0], input_maker(params))
                result = await gamla.to_awaitable(result)
                side_effect(params[0], result)
                return (
                    _wrap_in_result_if_needed(params[0], result),
                    _decisions_from_value_choices(params[2]),
                )

        else:

            def run_keeping_choices(params):
                result = _apply(params[0], input_maker(params))
                side_effect(params[0], result)
                return (
                    _wrap_in_result_if_needed(params[0], result),
                    _decisions_from_value_choices(params[2]),
                )

        return run_keeping_choices

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


def _dag_layer_reduce(
    f: Callable[
        [_IntermediaryResults, FrozenSet[base_types.ComputationNode]],
        _IntermediaryResults,
    ]
) -> Callable[[base_types.GraphType], _IntermediaryResults]:
    """Directed acyclic graph reduction."""
    return gamla.compose_left(
        graph.remove_future_edges,
        _toposort_nodes,
        gamla.reduce_curried(f, immutables.Map()),
    )


def _edge_to_value_options(
    accumulated_outputs,
) -> Callable[[Iterable[base_types.ComputationEdge]], Iterable[Any]]:
    return opt_gamla.mapduct(
        gamla.unless(
            gamla.attrgetter("is_future"),
            opt_gamla.compose_left(
                _get_edge_sources,
                opt_gamla.mapduct(
                    opt_gamla.compose_left(
                        opt_gamla.pair_left(
                            opt_gamla.compose_left(
                                gamla.dict_to_getter_with_default(
                                    immutables.Map(), accumulated_outputs
                                ),
                                collections.abc.Mapping.items,
                            )
                        ),
                        gamla.explode(0),
                    )
                ),
            ),
        )
    )


_create_node_run_options = opt_gamla.compose_left(
    gamla.pack,
    gamla.explode(1),
    opt_gamla.mapcat(
        opt_gamla.compose_left(
            gamla.bifurcate(
                gamla.head,
                gamla.second,
                opt_gamla.star(
                    lambda _, edges, edge_to_value_options: opt_gamla.pipe(
                        edges, graph.remove_future_edges, edge_to_value_options
                    )
                ),
            ),
            gamla.explode(2),
        )
    ),
)


def _assoc_immutable(d, k, v):
    return d.set(k, v)


@gamla.curry
def _lift_single_runner_to_run_on_many_options(is_async: bool, f):
    return (opt_async_gamla.compose_left if is_async else opt_gamla.compose_left)(
        _create_node_run_options,
        (opt_async_gamla.map if is_async else opt_gamla.map)(f),
        opt_gamla.filter(gamla.identity),
        dict,
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
                        _edge_to_value_options(accumulated_results),
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
                        _edge_to_value_options(accumulated_results),
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


def _make_runner(
    single_node_runner,
    is_async,
    async_decoration,
    edges,
    handled_exceptions,
    edges_to_node_id,
):
    return gamla.compose_left(
        # Higher order pipeline that constructs a graph runner.
        opt_gamla.compose(
            _dag_layer_reduce,
            _process_layer_in_parallel,
            _process_node(is_async, _incoming_edge_options(edges)),
            _lift_single_runner_to_run_on_many_options(is_async),
            gamla.excepts(
                (*handled_exceptions, _NotCoherent, base_types.SkipComputationError),
                opt_gamla.compose_left(type, _log_handled_exception, gamla.just(None)),
            ),
            single_node_runner,
            gamla.before(edges_to_node_id),
            _inject_state,
        ),
        # gamla.profileit,  # Enable to get a read on slow functions.
        # At this point we move to a regular pipeline of values.
        async_decoration(gamla.apply(edges)),
        gamla.attrgetter("__getitem__"),
    )


def _to_callable_with_side_effect_for_single_and_multiple(
    single_node_side_effect: _SingleNodeSideEffect,
    all_nodes_side_effect: Callable,
    edges: base_types.GraphType,
    handled_exceptions: FrozenSet[Type[Exception]],
) -> Callable:
    edges = gamla.pipe(edges, gamla.unique, tuple)
    edges_to_node_id = graph.edges_to_node_id_map(edges).__getitem__
    return gamla.compose_left(
        _make_outer_computation_input,
        gamla.pair_with(
            gamla.compose_left(
                _make_runner(
                    _run_keeping_choices(
                        _is_graph_async(edges), single_node_side_effect
                    ),
                    _is_graph_async(edges),
                    gamla.after(gamla.to_awaitable)
                    if _is_graph_async(edges)
                    else gamla.identity,
                    edges,
                    handled_exceptions,
                    edges_to_node_id,
                ),
                gamla.side_effect(all_nodes_side_effect(edges)),
                _construct_computation_result(edges, edges_to_node_id),
            )
        ),
        opt_gamla.star(
            lambda result_and_state, computation_input: _merge_with_previous_state(
                computation_input.state, *result_and_state
            )
        ),
    )


to_callable_with_side_effect = gamla.curry(
    _to_callable_with_side_effect_for_single_and_multiple
)(_type_check)

# Use the second line if you want to see the winning path in the computation graph (a little slower).
to_callable = to_callable_with_side_effect(gamla.just(gamla.just(None)))
# to_callable = to_callable_with_side_effect(graphviz.computation_trace('utterance_computation.dot'))


def _log_handled_exception(exception_type: Type[Exception]):
    _, exception, exception_traceback = sys.exc_info()
    filename, line_num, func_name, _ = traceback.extract_tb(exception_traceback)[-1]
    reason = ""
    if str(exception):
        reason = f": {exception}"
    code_location = f"{pathlib.Path(filename).name}:{line_num}"
    logging.debug(f"'{func_name.strip('_')}' {exception_type}@{code_location}{reason}")
