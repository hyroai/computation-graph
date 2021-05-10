import asyncio
import dataclasses
import itertools
import logging
import pathlib
import sys
import traceback
from typing import Any, Callable, Dict, FrozenSet, Set, Text, Tuple, Type

import gamla
import toposort

from computation_graph import base_types, graph, optimized_gamla

_get_edge_key = gamla.attrgetter("key")


class _ComputationGraphException(Exception):
    pass


def _transpose_graph(
    graph: Dict[base_types.ComputationNode, Set[base_types.ComputationNode]]
) -> Dict[base_types.ComputationNode, Set[base_types.ComputationNode]]:
    return optimized_gamla.pipe(
        graph.keys(),
        optimized_gamla.groupby_many(graph.get),
        optimized_gamla.valmap(set),
    )


def _get_edge_sources(edge: base_types.ComputationEdge):
    return edge.args or (edge.source,)


_toposort_nodes = optimized_gamla.compose_left(
    optimized_gamla.groupby_many(_get_edge_sources),
    optimized_gamla.valmap(
        optimized_gamla.compose_left(
            optimized_gamla.map(gamla.attrgetter("destination")), set
        )
    ),
    _transpose_graph,
    toposort.toposort,
    optimized_gamla.map(frozenset),
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


_incoming_edge_options = optimized_gamla.compose_left(
    graph.get_incoming_edges_for_node,
    gamla.after(
        optimized_gamla.compose_left(
            optimized_gamla.groupby(_get_edge_key),
            optimized_gamla.valmap(gamla.sort_by(gamla.attrgetter("priority"))),
            dict.values,
            optimized_gamla.star(itertools.product),
            optimized_gamla.map(tuple),
        )
    ),
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
        return optimized_gamla.pipe(
            edges_to_results,
            optimized_gamla.keyfilter(gamla.attrgetter("args")),
            dict.values,
            gamla.head,
            optimized_gamla.maptuple(gamla.attrgetter("result")),
        )
    return ()


def _get_unary_computation_input(
    kwargs: Tuple[Text, ...],
    value: base_types.ComputationResult,
    unbound_signature: base_types.NodeSignature,
) -> Dict[Text, Any]:
    return optimized_gamla.pipe(
        unbound_signature.kwargs,
        optimized_gamla.remove(
            optimized_gamla.anyjuxt(
                gamla.contains(unbound_signature.optional_kwargs), gamla.equals("state")
            )
        ),
        tuple,
        optimized_gamla.check(
            optimized_gamla.anyjuxt(gamla.len_equals(1), gamla.len_equals(0)),
            _ComputationGraphException(
                "got a single input function with more than 1 unbound arguments. cannot bind function"
            ),
        ),
        optimized_gamla.ternary(
            gamla.len_equals(1), gamla.identity, gamla.just(kwargs)
        ),
        gamla.head,
        lambda first_kwarg: {first_kwarg: value.result},
    )


def _get_outer_kwargs(
    unbound_signature: base_types.NodeSignature,
    unbound_input: base_types.ComputationInput,
) -> Dict[Text, Any]:
    # Optimized because being called a lot.
    d = {}
    for kwarg in unbound_signature.kwargs:
        if kwarg != "state" and kwarg in unbound_input.kwargs:
            d[kwarg] = unbound_input.kwargs[kwarg]
    return d


_get_inner_kwargs = optimized_gamla.compose_left(
    optimized_gamla.keyfilter(_get_edge_key),
    dict.items,
    optimized_gamla.groupby(optimized_gamla.compose_left(gamla.head, _get_edge_key)),
    optimized_gamla.valmap(
        optimized_gamla.compose_left(
            gamla.head, gamla.second, gamla.head, gamla.attrgetter("result")
        )
    ),
)


_DecisionsType = Dict[base_types.ComputationNode, base_types.ComputationResult]
_ResultToDecisionsType = Dict[base_types.ComputationResult, _DecisionsType]
_ResultToDependencies = Callable[[base_types.ComputationNode], _ResultToDecisionsType]


class _NotCoherent(Exception):
    """This exception signals that for a specific set of incoming
    node edges not all paths agree on the ComputationResult"""


class ComputationFailed(Exception):
    pass


def _check_equal_and_take_one(x, y):
    if x == y:
        return x
    raise _NotCoherent


NodeToResults = Callable[[base_types.ComputationNode], _ResultToDecisionsType]


def _node_to_value_choices(node_to_results: NodeToResults):
    return optimized_gamla.compose_left(
        optimized_gamla.pair_left(
            optimized_gamla.compose_left(node_to_results, dict.items)
        ),
        optimized_gamla.packstack(gamla.identity, itertools.repeat),
        optimized_gamla.star(zip),
    )


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


def _get_computation_input(
    unbound_input: base_types.ComputationInput,
    signature: base_types.NodeSignature,
    incoming_edges: base_types.GraphType,
    results: Tuple[Tuple[base_types.ComputationResult, ...], ...],
) -> base_types.ComputationInput:
    bound_signature = base_types.NodeSignature(
        is_args=signature.is_args and any(edge.args for edge in incoming_edges),
        kwargs=optimized_gamla.pipe(
            incoming_edges,
            optimized_gamla.map(_get_edge_key),
            optimized_gamla.remove(gamla.equals(None)),
            tuple,
        ),
    )
    unbound_signature = _signature_difference(signature, bound_signature)
    if (
        not (unbound_signature.is_args or bound_signature.is_args)
        and sum(
            map(
                optimized_gamla.compose_left(_get_edge_key, gamla.equals(None)),
                incoming_edges,
            )
        )
        == 1
    ):
        return base_types.ComputationInput(
            args=(),
            kwargs=_get_unary_computation_input(
                signature.kwargs, gamla.head(gamla.head(results)), unbound_signature
            ),
            state=unbound_input.state,
        )
    edges_to_results = dict(zip(incoming_edges, results))
    return base_types.ComputationInput(
        args=_get_args(
            edges_to_results, unbound_signature, bound_signature, unbound_input
        ),
        kwargs={
            **_get_outer_kwargs(unbound_signature, unbound_input),
            **_get_inner_kwargs(edges_to_results),
        },
        state=unbound_input.state,
    )


def _wrap_in_result_if_needed(result):
    if isinstance(result, base_types.ComputationResult):
        return result
    return base_types.ComputationResult(result=result, state=None)


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


_choice_to_value = optimized_gamla.compose_left(gamla.head, gamla.head)

_decisions_from_value_choices = optimized_gamla.compose_left(
    gamla.concat,
    gamla.bifurcate(
        optimized_gamla.compose_left(
            optimized_gamla.map(optimized_gamla.compose_left(gamla.head, gamla.second)),
            optimized_gamla.reduce(
                optimized_gamla.merge_with_reducer(_check_equal_and_take_one),
                gamla.frozendict(),
            ),
        ),
        optimized_gamla.mapdict(optimized_gamla.juxt(gamla.second, _choice_to_value)),
    ),
    optimized_gamla.merge,
)


def _construct_computation_state(
    results: _ResultToDecisionsType, sink_node: base_types.ComputationNode
) -> Dict:
    first_result = gamla.head(results)
    return {
        **{sink_node: first_result.state},
        **optimized_gamla.pipe(
            results,
            gamla.itemgetter(first_result),
            optimized_gamla.valmap(gamla.attrgetter("state")),
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


def _construct_computation_result(edges: base_types.GraphType, edges_to_node_id):
    def construct_computation_result(result_to_dependencies: _ResultToDependencies):
        return optimized_gamla.pipe(
            edges,
            graph.infer_graph_sink,
            optimized_gamla.pair_left(
                gamla.translate_exception(
                    optimized_gamla.compose_left(result_to_dependencies, gamla.head),
                    (StopIteration, KeyError),
                    ComputationFailed,
                )
            ),
            optimized_gamla.check(gamla.identity, ComputationFailed),
            gamla.stack(
                (
                    gamla.attrgetter("result"),
                    optimized_gamla.compose_left(
                        optimized_gamla.pair_left(result_to_dependencies),
                        optimized_gamla.star(_construct_computation_state),
                        optimized_gamla.keymap(edges_to_node_id),
                    ),
                )
            ),
        )

    return construct_computation_result


def _apply(node: base_types.ComputationNode, node_input: base_types.ComputationInput):
    return node.func(
        *node_input.args,
        **gamla.add_key_value("state", node_input.state)(node_input.kwargs)
        if node.is_stateful
        else node_input.kwargs,
    )


def _run_keeping_choices(is_async: bool):
    def run_keeping_choices(node_to_external_input):
        if is_async:

            async def run_keeping_choices(params):
                node, edges_choice, values_for_edges_choice = params
                return (
                    _wrap_in_result_if_needed(
                        await gamla.to_awaitable(
                            _apply(
                                node,
                                _get_computation_input(
                                    node_to_external_input(node),
                                    node.signature,
                                    edges_choice,
                                    optimized_gamla.maptuple(
                                        optimized_gamla.maptuple(_choice_to_value)
                                    )(values_for_edges_choice),
                                ),
                            )
                        )
                    ),
                    _decisions_from_value_choices(values_for_edges_choice),
                )

            return run_keeping_choices

        def run_keeping_choices_sync(params):
            node, edges_choice, values_for_edges_choice = params
            return (
                _wrap_in_result_if_needed(
                    _apply(
                        node,
                        _get_computation_input(
                            node_to_external_input(node),
                            node.signature,
                            edges_choice,
                            optimized_gamla.maptuple(
                                optimized_gamla.maptuple(_choice_to_value)
                            )(values_for_edges_choice),
                        ),
                    )
                ),
                _decisions_from_value_choices(values_for_edges_choice),
            )

        return run_keeping_choices_sync

    return run_keeping_choices


def _process_layer_in_parallel(f):
    return gamla.compose_left(
        gamla.pack, gamla.explode(1), gamla.map(f), optimized_gamla.merge
    )


def _dag_layer_reduce(f: Callable):
    """Directed acyclic graph reduction."""
    return gamla.compose_left(_toposort_nodes, gamla.reduce_curried(f, {}))


def _edge_to_value_options(accumulated_outputs):
    return optimized_gamla.mapduct(
        optimized_gamla.compose_left(
            _get_edge_sources,
            optimized_gamla.mapduct(
                _node_to_value_choices(
                    gamla.excepts(
                        KeyError,
                        gamla.just(gamla.frozendict()),
                        accumulated_outputs.__getitem__,
                    )
                )
            ),
        )
    )


_bla = optimized_gamla.compose_left(
    gamla.explode(1),
    optimized_gamla.mapcat(
        optimized_gamla.juxtduct(
            optimized_gamla.compose_left(gamla.head, gamla.wrap_tuple),
            optimized_gamla.compose_left(gamla.second, gamla.wrap_tuple),
            optimized_gamla.star(lambda _, y, z: z(y)),
        )
    ),
)


def _process_node(is_async, get_edge_options):
    def process_node(f):
        if is_async:

            async def process_node(params):
                accumulated_results, node = params
                params = (
                    node,
                    get_edge_options(node),
                    _edge_to_value_options(accumulated_results),
                )
                accumulated_results[node] = await optimized_gamla.compose_left_async(
                    _bla,
                    optimized_gamla.map_async(f),
                    optimized_gamla.filter(gamla.identity),
                    dict,
                )(params)

                return accumulated_results

            return process_node

        def process_node_sync(params):
            accumulated_results, node = params
            params = (
                node,
                get_edge_options(node),
                _edge_to_value_options(accumulated_results),
            )
            accumulated_results[node] = optimized_gamla.compose_left(
                _bla,
                optimized_gamla.map(f),
                optimized_gamla.filter(gamla.identity),
                dict,
            )(params)

            return accumulated_results

        return process_node_sync

    return process_node


_is_graph_async = optimized_gamla.compose_left(
    optimized_gamla.mapcat(lambda edge: (edge.source, *edge.args)),
    optimized_gamla.remove(gamla.equals(None)),
    optimized_gamla.map(gamla.attrgetter("func")),
    gamla.anymap(asyncio.iscoroutinefunction),
)


def _make_runner(
    is_async, side_effect, async_decoration, edges, handled_exceptions, edges_to_node_id
):
    return gamla.compose_left(
        # Higher order pipeline that constructs a graph runner.
        optimized_gamla.compose(
            _dag_layer_reduce,
            _process_layer_in_parallel,
            _process_node(is_async, _incoming_edge_options(edges)),
            gamla.excepts(
                (*handled_exceptions, _NotCoherent),
                optimized_gamla.compose_left(
                    type, _log_handled_exception, gamla.just(None)
                ),
            ),
            _run_keeping_choices(is_async),
            gamla.before(edges_to_node_id),
            _inject_state,
        ),
        # At this point we move to a regular pipeline of values.
        async_decoration(gamla.apply(edges)),
        gamla.attrgetter("__getitem__"),
        gamla.side_effect(side_effect),
        _construct_computation_result(edges, edges_to_node_id),
    )


def to_callable_with_side_effect(
    side_effect: Callable,
    edges: base_types.GraphType,
    handled_exceptions: FrozenSet[Type[Exception]],
) -> Callable:
    edges = tuple(gamla.unique(edges))
    return gamla.compose_left(
        _make_outer_computation_input,
        gamla.pair_with(
            _make_runner(
                _is_graph_async(edges),
                side_effect(edges),
                gamla.after(gamla.to_awaitable)
                if _is_graph_async(edges)
                else gamla.identity,
                edges,
                handled_exceptions,
                graph.edges_to_node_id_map(edges).__getitem__,
            )
        ),
        optimized_gamla.star(
            lambda result_and_state, computation_input: _merge_with_previous_state(
                computation_input.state, *result_and_state
            )
        ),
    )


to_callable = gamla.curry(to_callable_with_side_effect)(gamla.just(gamla.just(None)))


def _log_handled_exception(exception_type: Type[Exception]):
    _, exception, exception_traceback = sys.exc_info()
    filename, line_num, func_name, _ = traceback.extract_tb(exception_traceback)[-1]
    reason = ""
    if str(exception):
        reason = f": {exception}"
    code_location = f"{pathlib.Path(filename).name}:{line_num}"
    logging.debug(f"'{func_name.strip('_')}' {exception_type}@{code_location}{reason}")
