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

from computation_graph import base_types, graph

_get_edge_key = gamla.attrgetter("key")


class _ComputationGraphException(Exception):
    pass


def _transpose_graph(
    graph: Dict[base_types.ComputationNode, Set[base_types.ComputationNode]]
) -> Dict[base_types.ComputationNode, Set[base_types.ComputationNode]]:
    return gamla.pipe(graph.keys(), gamla.groupby_many(graph.get), gamla.valmap(set))


def _get_edge_sources(edge: base_types.ComputationEdge):
    return edge.args or (edge.source,)


_toposort_nodes = gamla.compose_left(
    gamla.groupby_many(_get_edge_sources),
    gamla.valmap(gamla.compose_left(gamla.map(gamla.attrgetter("destination")), set)),
    _transpose_graph,
    toposort.toposort,
    gamla.map(frozenset),
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


_incoming_edge_options = gamla.compose_left(
    graph.get_incoming_edges_for_node,
    gamla.after(
        gamla.compose_left(
            gamla.groupby(_get_edge_key),
            gamla.valmap(gamla.sort_by(gamla.attrgetter("priority"))),
            dict.values,
            gamla.star(itertools.product),
            gamla.map(tuple),
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
        return gamla.pipe(
            edges_to_results,
            gamla.keyfilter(gamla.attrgetter("args")),
            dict.values,
            gamla.head,
            _maptuple(gamla.attrgetter("result")),
        )
    return ()


def _get_unary_computation_input(
    kwargs: Tuple[Text, ...],
    value: base_types.ComputationResult,
    unbound_signature: base_types.NodeSignature,
) -> Dict[Text, Any]:
    return gamla.pipe(
        unbound_signature.kwargs,
        gamla.remove(
            gamla.anyjuxt(
                gamla.contains(unbound_signature.optional_kwargs), gamla.equals("state")
            )
        ),
        tuple,
        gamla.check(
            gamla.anyjuxt(gamla.len_equals(1), gamla.len_equals(0)),
            _ComputationGraphException(
                "got a single input function with more than 1 unbound arguments. cannot bind function"
            ),
        ),
        gamla.ternary(gamla.len_equals(1), gamla.identity, gamla.just(kwargs)),
        gamla.head,
        lambda first_kwarg: {first_kwarg: value.result},
    )


def _get_outer_kwargs(
    unbound_signature: base_types.NodeSignature,
    unbound_input: base_types.ComputationInput,
) -> Dict[Text, Any]:
    return gamla.pipe(
        unbound_signature.kwargs,
        gamla.remove(gamla.equals("state")),
        gamla.map(gamla.pair_right(unbound_input.kwargs.get)),
        gamla.remove(gamla.compose_left(gamla.second, gamla.equals(None))),
        dict,
    )


_get_inner_kwargs = gamla.compose_left(
    gamla.keyfilter(_get_edge_key),
    dict.items,
    gamla.groupby(gamla.compose_left(gamla.head, _get_edge_key)),
    gamla.valmap(
        gamla.compose_left(
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


_merge_decision = gamla.merge_with(
    gamla.compose_left(
        gamla.unique, tuple, gamla.check(gamla.len_equals(1), _NotCoherent), gamla.head
    )
)

NodeToResults = Callable[[base_types.ComputationNode], _ResultToDecisionsType]


def _node_to_value_choices(node_to_results: NodeToResults):
    return gamla.compose_left(
        gamla.pair_with(gamla.compose_left(node_to_results, dict.items)),
        gamla.stack([gamla.identity, itertools.repeat]),
        gamla.star(zip),
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
        kwargs=gamla.pipe(
            incoming_edges,
            gamla.map(_get_edge_key),
            gamla.remove(gamla.equals(None)),
            tuple,
        ),
    )
    unbound_signature = _signature_difference(signature, bound_signature)
    if (
        not (unbound_signature.is_args or bound_signature.is_args)
        and sum(
            map(gamla.compose_left(_get_edge_key, gamla.equals(None)), incoming_edges)
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
        kwargs=gamla.merge(
            _get_outer_kwargs(unbound_signature, unbound_input),
            _get_inner_kwargs(edges_to_results),
        ),
        state=unbound_input.state,
    )


def _wrap_in_result_if_needed(result):
    if isinstance(result, base_types.ComputationResult):
        return result
    return base_types.ComputationResult(result=result, state=None)


@gamla.curry
def _inject_state(
    unbound_input: base_types.ComputationInput, node_id: int
) -> base_types.ComputationInput:
    if unbound_input.state is None:
        return unbound_input
    return dataclasses.replace(
        unbound_input,
        state=unbound_input.state[node_id] if node_id in unbound_input.state else None,
    )


_juxtduct = gamla.compose_left(gamla.juxt, gamla.after(gamla.star(itertools.product)))
_mapdict = gamla.compose_left(gamla.map, gamla.after(dict))
_mapduct = gamla.compose_left(gamla.map, gamla.after(gamla.star(itertools.product)))
_maptuple = gamla.compose_left(gamla.map, gamla.after(tuple))

_choice_to_value = gamla.compose_left(gamla.head, gamla.head)

_decisions_from_value_choices = gamla.compose_left(
    gamla.concat,
    gamla.bifurcate(
        gamla.compose_left(
            gamla.map(gamla.compose_left(gamla.head, gamla.second)),
            gamla.reduce(_merge_decision, gamla.frozendict()),
        ),
        _mapdict(gamla.juxt(gamla.second, _choice_to_value)),
    ),
    gamla.merge,
)


def _construct_computation_state(
    results: _ResultToDecisionsType, sink_node: base_types.ComputationNode
) -> Dict:
    first_result = gamla.head(results)
    return gamla.merge(
        {sink_node: first_result.state},
        gamla.pipe(
            results,
            gamla.itemgetter(first_result),
            gamla.valmap(gamla.attrgetter("state")),
        ),
    )


def _merge_with_previous_state(
    previous_state: Dict, result: base_types.ComputationResult, state: Dict
) -> base_types.ComputationResult:
    return base_types.ComputationResult(
        result=result,
        # Convert to tuples (node id, state) so this would be hashable.
        state=tuple(gamla.merge(previous_state or {}, state).items()),
    )


@gamla.curry
def _construct_computation_result(
    edges: base_types.GraphType,
    edges_to_node_id,
    result_to_dependencies: _ResultToDependencies,
) -> Tuple[base_types.ComputationResult, Dict]:
    return gamla.pipe(
        edges,
        graph.infer_graph_sink,
        gamla.pair_with(
            gamla.translate_exception(
                gamla.compose_left(result_to_dependencies, gamla.head),
                (StopIteration, KeyError),
                ComputationFailed,
            )
        ),
        gamla.check(gamla.identity, ComputationFailed),
        gamla.stack(
            (
                gamla.attrgetter("result"),
                gamla.compose_left(
                    gamla.pair_with(result_to_dependencies),
                    gamla.star(_construct_computation_state),
                    gamla.keymap(edges_to_node_id),
                ),
            )
        ),
    )


def _apply(node: base_types.ComputationNode, node_input: base_types.ComputationInput):
    return node.func(
        *node_input.args,
        **gamla.add_key_value("state", node_input.state)(node_input.kwargs)
        if node.is_stateful
        else node_input.kwargs,
    )


@gamla.curry
def _run_keeping_choices(
    run_node_for_input: Callable[
        [base_types.ComputationNode, base_types.ComputationInput],
        base_types.ComputationResult,
    ],
    node_to_external_input: Callable[
        [base_types.ComputationNode], base_types.ComputationInput,
    ],
):
    return gamla.juxt(
        gamla.compose_many_to_one(
            [
                gamla.head,  # node
                gamla.compose_many_to_one(  # computation input
                    [
                        gamla.compose_left(gamla.head, node_to_external_input),
                        gamla.compose_left(gamla.head, gamla.attrgetter("signature")),
                        gamla.second,  # edges choice
                        gamla.compose_left(  # dependencies
                            gamla.nth(2),  # values for edges choice
                            _maptuple(_maptuple(_choice_to_value)),
                        ),
                    ],
                    _get_computation_input,
                ),
            ],
            run_node_for_input,
        ),
        gamla.compose_left(gamla.nth(2), _decisions_from_value_choices),
    )


def _process_layer_in_parallel(f):
    return gamla.compose_left(
        lambda state, layer: ((state,), layer),
        gamla.star(itertools.product),
        gamla.map(f),
        gamla.merge,
    )


def _dag_layer_reduce(f: Callable):
    """Directed acyclic graph reduction."""
    return gamla.compose_left(_toposort_nodes, gamla.reduce_curried(f, {}))


def _edge_to_value_options(accumulated_outputs):
    return _mapduct(
        gamla.compose_left(
            _get_edge_sources,
            _mapduct(
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


@gamla.curry
def _process_node(get_edge_options, f):
    return gamla.compose_many_to_one(
        [
            gamla.head,  # accumulated results
            gamla.compose_left(gamla.second, gamla.wrap_tuple),  # node
            gamla.compose_left(  # new results
                _juxtduct(
                    gamla.compose_left(gamla.second, gamla.wrap_tuple),
                    gamla.compose_left(gamla.second, get_edge_options),
                    gamla.compose_left(
                        gamla.head, _edge_to_value_options, gamla.wrap_tuple
                    ),
                ),
                gamla.mapcat(
                    _juxtduct(
                        gamla.compose_left(gamla.head, gamla.wrap_tuple),
                        gamla.compose_left(gamla.second, gamla.wrap_tuple),
                        gamla.star(lambda _, y, z: z(y)),
                    )
                ),
                gamla.map(f),
                gamla.filter(gamla.identity),
                dict,
            ),
        ],
        gamla.assoc_in,
    )


_is_graph_async = gamla.compose_left(
    gamla.mapcat(lambda edge: (edge.source, *edge.args)),
    gamla.remove(gamla.equals(None)),
    gamla.map(gamla.attrgetter("func")),
    gamla.anymap(asyncio.iscoroutinefunction),
)


def _make_runner(
    side_effect, async_decoration, edges, handled_exceptions, edges_to_node_id
):
    return gamla.compose_left(
        # Higher order pipeline that constructs a graph runner.
        gamla.compose(
            _dag_layer_reduce,
            _process_layer_in_parallel,
            _process_node(_incoming_edge_options(edges)),
            gamla.excepts(
                (*handled_exceptions, _NotCoherent),
                gamla.compose_left(type, _log_handled_exception, gamla.just(None)),
            ),
            _run_keeping_choices(
                gamla.compose_left(async_decoration(_apply), _wrap_in_result_if_needed)
            ),
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
    return gamla.compose_left(
        _make_outer_computation_input,
        gamla.pair_with(
            _make_runner(
                side_effect(edges),
                gamla.after(gamla.to_awaitable)
                if _is_graph_async(edges)
                else gamla.identity,
                edges,
                handled_exceptions,
                graph.edges_to_node_id_map(edges).__getitem__,
            )
        ),
        gamla.star(
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
