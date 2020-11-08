import asyncio
import dataclasses
import itertools
import logging
import pathlib
import sys
import traceback
from typing import Any, Callable, Dict, FrozenSet, Set, Text, Tuple, Type

import gamla
import toolz
import toposort
from toolz import curried

from computation_graph import base_types, config, graph

COMPUTATION_TRACE_DOT_FILENAME = "computation.dot"

_get_edge_key = gamla.attrgetter("key")


class _ComputationGraphException(Exception):
    pass


def _transpose_graph(
    graph: Dict[base_types.ComputationNode, Set[base_types.ComputationNode]],
) -> Dict[base_types.ComputationNode, Set[base_types.ComputationNode]]:
    return gamla.pipe(graph.keys(), gamla.groupby_many(graph.get), gamla.valmap(set))


def _get_edge_sources(edge: base_types.ComputationEdge):
    return edge.args or (edge.source,)


_toposort_nodes = gamla.compose_left(
    gamla.groupby_many(_get_edge_sources),
    gamla.valmap(
        gamla.compose_left(gamla.map(lambda edge: edge.destination), set),
    ),
    _transpose_graph,
    toposort.toposort,
    gamla.map(frozenset),
    tuple,
)


def _make_computation_input(*args, **kwargs) -> base_types.ComputationInput:
    if "state" in kwargs:
        return base_types.ComputationInput(
            args=args,
            kwargs=toolz.dissoc(kwargs, "state"),
            state=dict(kwargs["state"] or {}),
        )

    return base_types.ComputationInput(args=args, kwargs=kwargs)


_incoming_edge_options = gamla.compose_left(
    graph.get_incoming_edges_for_node,
    gamla.after(
        gamla.compose_left(
            curried.groupby(_get_edge_key),
            gamla.valmap(curried.sorted(key=lambda edge: edge.priority)),
            dict.values,
            gamla.star(itertools.product),
            gamla.map(tuple),
        ),
    ),
)


def _get_args(
    edges_to_results,
    unbound_signature: base_types.NodeSignature,
    bound_signature: base_types.NodeSignature,
    unbound_input: base_types.ComputationInput,
) -> Tuple[base_types.ComputationResult, ...]:
    if unbound_signature.is_args:
        return unbound_input.args

    if bound_signature.is_args:
        return gamla.pipe(
            edges_to_results,
            curried.keyfilter(gamla.attrgetter("args")),
            dict.values,
            toolz.first,
            _maptuple(gamla.attrgetter("result")),
        )

    return ()


def _get_unary_computation_input(
    kwargs,
    value: base_types.ComputationResult,
    unbound_signature: base_types.NodeSignature,
) -> Dict[Text, Any]:
    return gamla.pipe(
        unbound_signature.kwargs,
        gamla.filter(
            lambda arg: arg not in unbound_signature.optional_kwargs and arg != "state",
        ),
        tuple,
        gamla.check(
            gamla.anyjuxt(gamla.len_equals(1), gamla.len_equals(0)),
            _ComputationGraphException(
                "got a single input function with more than 1 unbound arguments. cannot bind function",
            ),
        ),
        gamla.ternary(gamla.len_equals(1), gamla.identity, lambda _: kwargs),
        toolz.first,
        lambda first_kwarg: {first_kwarg: value.result},
    )


def _get_kwargs(
    edges_to_results,
    unbound_signature: base_types.NodeSignature,
    unbound_input: base_types.ComputationInput,
) -> Dict[Text, Any]:
    kwargs = gamla.pipe(
        unbound_signature.kwargs,
        gamla.filter(lambda arg: arg != "state"),
        gamla.map(gamla.pair_right(unbound_input.kwargs.get)),
        gamla.filter(gamla.star(lambda _, value: value is not None)),
        dict,
    )

    return gamla.pipe(
        edges_to_results,
        curried.keyfilter(_get_edge_key),
        dict.items,
        curried.groupby(gamla.compose_left(toolz.first, _get_edge_key)),
        gamla.valmap(
            gamla.compose_left(
                toolz.first,
                toolz.second,
                toolz.first,
                gamla.attrgetter("result"),
            ),
        ),
        lambda x: (kwargs, x),
        toolz.merge,
    )


_DecisionsType = Dict[base_types.ComputationNode, base_types.ComputationResult]
_ResultToDecisionsType = Dict[base_types.ComputationResult, _DecisionsType]
_ResultToDependencies = Callable[[base_types.ComputationNode], _ResultToDecisionsType]


class _NotCoherent(Exception):
    """This exception signals that for a specific set of incoming
    node edges not all paths agree on the ComputationResult"""


class ComputationFailed(Exception):
    pass


_merge_decision = curried.merge_with(
    gamla.compose_left(
        toolz.unique,
        tuple,
        gamla.check(gamla.len_equals(1), _NotCoherent),
        toolz.first,
    ),
)

_NodeToResults = Callable[[base_types.ComputationNode], _ResultToDecisionsType]


def _node_to_value_choices(node_to_results: _NodeToResults):
    return gamla.compose_left(
        gamla.pair_with(gamla.compose_left(node_to_results, dict.items)),
        gamla.stack([toolz.identity, itertools.repeat]),
        gamla.star(zip),
    )


def _map_product(f):
    return gamla.compose_left(gamla.map(f), gamla.star(itertools.product))


def _signature_difference(
    sig_a: base_types.NodeSignature,
    sig_b: base_types.NodeSignature,
) -> base_types.NodeSignature:
    return base_types.NodeSignature(
        is_args=(sig_a.is_args != sig_b.is_args),
        # Difference must save the order of the left signature.
        kwargs=tuple(filter(lambda x: x not in sig_b.kwargs, sig_a.kwargs)),
        optional_kwargs=tuple(
            filter(lambda x: x not in sig_b.optional_kwargs, sig_a.optional_kwargs),
        ),
    )


@gamla.curry
def _get_computation_input(
    get_unbound_input_for_node: Callable[
        [base_types.ComputationNode],
        base_types.ComputationInput,
    ],
    node: base_types.ComputationNode,
    incoming_edges_for_node: base_types.GraphType,
    results: Tuple[Tuple[base_types.ComputationResult, ...], ...],
) -> base_types.ComputationInput:
    bound_signature = base_types.NodeSignature(
        is_args=node.signature.is_args
        and any(edge.args for edge in incoming_edges_for_node),
        kwargs=tuple(filter(None, map(_get_edge_key, incoming_edges_for_node))),
    )
    unbound_signature = _signature_difference(node.signature, bound_signature)
    unbound_input_for_node = get_unbound_input_for_node(node)

    if (
        not (unbound_signature.is_args or bound_signature.is_args)
        and sum(
            map(
                gamla.compose_left(_get_edge_key, gamla.equals(None)),
                incoming_edges_for_node,
            ),
        )
        == 1
    ):
        return base_types.ComputationInput(
            args=(),
            kwargs=_get_unary_computation_input(
                node.signature.kwargs,
                toolz.first(toolz.first(results)),
                unbound_signature,
            ),
            state=unbound_input_for_node.state,
        )

    edges_to_results = dict(zip(incoming_edges_for_node, results))
    return base_types.ComputationInput(
        args=_get_args(
            edges_to_results,
            unbound_signature,
            bound_signature,
            unbound_input_for_node,
        ),
        kwargs=_get_kwargs(edges_to_results, unbound_signature, unbound_input_for_node),
        state=unbound_input_for_node.state,
    )


def _wrap_in_result_if_needed(result):
    if isinstance(result, base_types.ComputationResult):
        return result
    return base_types.ComputationResult(result=result, state=None)


@gamla.curry
def _get_node_unbound_input(
    unbound_input: base_types.ComputationInput,
    node_id: int,
) -> base_types.ComputationInput:
    if unbound_input.state is None:
        return unbound_input
    return dataclasses.replace(
        unbound_input,
        state=unbound_input.state[node_id] if node_id in unbound_input.state else None,
    )


def _maptuple(f):
    return gamla.compose_left(gamla.map(f), tuple)


def _mapdict(f):
    return gamla.compose_left(gamla.map(f), dict)


_choice_to_value = gamla.compose_left(toolz.first, toolz.first)

_decisions_from_value_choices = gamla.compose_left(
    toolz.concat,
    gamla.bifurcate(
        gamla.compose_left(
            gamla.map(gamla.compose_left(toolz.first, toolz.second)),
            gamla.reduce(_merge_decision, gamla.frozendict()),
        ),
        _mapdict(gamla.juxt(toolz.second, _choice_to_value)),
    ),
    toolz.merge,
)


def _construct_computation_state(
    results: _ResultToDecisionsType,
    sink_node: base_types.ComputationNode,
) -> Dict:
    first_result = toolz.first(results)
    return toolz.merge(
        {sink_node: first_result.state},
        gamla.pipe(
            results,
            gamla.itemgetter(first_result),
            gamla.valmap(gamla.attrgetter("state")),
        ),
    )


def _merge_with_previous_state(
    previous_state: Dict,
    result: base_types.ComputationResult,
    state: Dict,
) -> base_types.ComputationResult:
    return base_types.ComputationResult(
        result=result,
        # Convert to tuples (node id, state) so this would be hashable.
        state=tuple(toolz.merge(previous_state or {}, state).items()),
    )


@gamla.curry
def _construct_computation_result(
    edges: base_types.GraphType,
    result_to_dependencies: _ResultToDependencies,
):
    return gamla.pipe(
        edges,
        graph.infer_graph_sink,
        _debug_trace(edges, _node_computation_trace(result_to_dependencies)),
        gamla.pair_with(
            gamla.translate_exception(
                gamla.compose_left(result_to_dependencies, toolz.first),
                (StopIteration, KeyError),
                ComputationFailed,
            ),
        ),
        gamla.check(gamla.identity, ComputationFailed),
        gamla.stack(
            (
                gamla.attrgetter("result"),
                gamla.compose_left(
                    gamla.pair_with(result_to_dependencies),
                    gamla.star(_construct_computation_state),
                    gamla.keymap(graph.infer_node_id(edges)),
                ),
            ),
        ),
    )


_ComputationTrace = Callable[
    [base_types.ComputationNode],
    Tuple[Tuple[base_types.ComputationNode, base_types.ComputationResult]],
]


def _debug_trace(
    edges: base_types.GraphType,
    computation_trace: _ComputationTrace,
) -> Callable[[base_types.ComputationNode], Any]:
    if config.DEBUG_SAVE_COMPUTATION_TRACE:
        from computation_graph import visualization

        return curried.do(
            gamla.compose_left(
                computation_trace,
                gamla.pair_with(gamla.just(edges)),
                visualization.serialize_computation_trace(
                    COMPUTATION_TRACE_DOT_FILENAME,
                ),
            ),
        )

    return gamla.identity


def _node_computation_trace(node_to_results: _NodeToResults) -> _ComputationTrace:
    return gamla.compose_left(
        gamla.juxt(
            gamla.pair_right(
                gamla.compose_left(node_to_results, dict.keys, toolz.first),
            ),
            gamla.compose_left(node_to_results, dict.values, toolz.first, dict.items),
        ),
        gamla.star(toolz.cons),
    )


def _apply(node, node_input):
    return node.func(
        *node_input.args,
        **toolz.assoc(node_input.kwargs, "state", node_input.state)
        if node.is_stateful
        else node_input.kwargs,
    )


@gamla.curry
def _run_keeping_choices(apply, make_input):
    return gamla.juxt(
        gamla.compose_left(
            gamla.juxt(
                toolz.first,
                gamla.compose_left(
                    gamla.stack(
                        [
                            toolz.identity,
                            toolz.identity,
                            _maptuple(_maptuple(_choice_to_value)),
                        ],
                    ),
                    gamla.star(make_input),
                ),
            ),
            gamla.star(apply),
            _wrap_in_result_if_needed,
        ),
        gamla.compose_left(curried.nth(2), _decisions_from_value_choices),
    )


def _process_layer_in_parallel(f):
    return gamla.compose_left(
        lambda state, layer: ((state,), layer),
        gamla.star(itertools.product),
        gamla.map(f),
        toolz.merge,
    )


def _dag_layer_reduce(f: Callable):
    """Directed acyclic graph reduction."""
    return gamla.compose_left(
        _toposort_nodes,
        gamla.reduce_curried(f, {}),
    )


def _per_values_option(f):
    return gamla.compose_left(
        gamla.star(
            lambda accumulated_outputs, node, edge_option: (
                (node,),
                (edge_option,),
                gamla.pipe(
                    edge_option,
                    _map_product(
                        gamla.compose_left(
                            _get_edge_sources,
                            _map_product(
                                _node_to_value_choices(
                                    gamla.excepts(
                                        KeyError,
                                        gamla.just(gamla.frozendict()),
                                        accumulated_outputs.__getitem__,
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
        gamla.star(itertools.product),
        gamla.map(f),
        gamla.filter(gamla.identity),
        dict,
    )


@gamla.curry
def _per_edge_option(get_edge_options, f):
    return gamla.compose_left(
        gamla.juxt(
            toolz.first,
            toolz.second,
            gamla.compose_left(
                gamla.juxt(
                    gamla.compose_left(toolz.first, itertools.repeat),
                    gamla.compose_left(toolz.second, itertools.repeat),
                    gamla.compose_left(toolz.second, get_edge_options),
                ),
                gamla.star(zip),
                gamla.map(f),
                toolz.merge,
            ),
        ),
        gamla.star(curried.assoc),
    )


_is_graph_async = gamla.compose_left(
    curried.mapcat(lambda edge: (edge.source, *edge.args)),
    gamla.filter(gamla.identity),
    gamla.map(lambda node: node.func),
    gamla.anymap(asyncio.iscoroutinefunction),
)


def to_callable(
    edges: base_types.GraphType,
    handled_exceptions: FrozenSet[Type[Exception]],
) -> Callable:
    return gamla.compose_left(
        _make_computation_input,
        gamla.pair_with(
            # Note: this is a higher order pipeline which builds a function, until the call to `apply`.
            gamla.compose_left(
                _get_node_unbound_input,
                gamla.before(graph.infer_node_id(edges)),
                _get_computation_input,
                _run_keeping_choices(
                    gamla.compose(gamla.to_awaitable, _apply)
                    if _is_graph_async(edges)
                    else _apply,
                ),
                gamla.excepts(
                    (
                        *handled_exceptions,
                        _NotCoherent,
                    ),
                    gamla.compose_left(
                        type,
                        _log_handled_exception,
                        gamla.just(None),
                    ),
                ),
                _per_values_option,
                _per_edge_option(_incoming_edge_options(edges)),
                _process_layer_in_parallel,
                _dag_layer_reduce,
                gamla.apply(edges),
                gamla.to_awaitable if _is_graph_async(edges) else gamla.identity,
                gamla.attrgetter("__getitem__"),
                _construct_computation_result(edges),
            ),
        ),
        gamla.star(
            lambda result_and_state, computation_input: _merge_with_previous_state(
                computation_input.state, *result_and_state
            ),
        ),
    )


def _log_handled_exception(exception_type: Type[Exception]):
    _, exception, exception_traceback = sys.exc_info()
    filename, line_num, func_name, _ = traceback.extract_tb(exception_traceback)[-1]
    if str(exception):
        reason = f": {exception}"
    else:
        reason = ""
    code_location = f"{pathlib.Path(filename).name}:{line_num}"
    logging.debug(f"'{func_name.strip('_')}' {exception_type}@{code_location}{reason}")
