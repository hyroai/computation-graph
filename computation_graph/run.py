import dataclasses
import itertools
import logging
import pathlib
import sys
import traceback
from typing import Any, Callable, Dict, FrozenSet, Optional, Set, Text, Tuple, Type

import gamla
import toolz
import toposort
from toolz import curried

from computation_graph import base_types, config, graph

COMPUTATION_TRACE_DOT_FILENAME = "computation.dot"


class _ComputationGraphException(Exception):
    pass


def _transpose_graph(
    graph: Dict[base_types.ComputationNode, Set[base_types.ComputationNode]],
) -> Dict[base_types.ComputationNode, Set[base_types.ComputationNode]]:
    return toolz.pipe(graph.keys(), gamla.groupby_many(graph.get), curried.valmap(set))


def _toposort_nodes(
    edges: base_types.GraphType,
) -> Tuple[FrozenSet[base_types.ComputationNode], ...]:
    return toolz.pipe(
        edges,
        gamla.groupby_many(lambda edge: edge.args or (edge.source,)),
        curried.valmap(
            toolz.compose_left(curried.map(lambda edge: edge.destination), set),
        ),
        _transpose_graph,
        toposort.toposort,
        curried.map(frozenset),
        tuple,
    )


def _result_from_computation_result(
    computation_result: base_types.ComputationResult,
) -> base_types.ComputationResult:
    return computation_result.result


def _make_computation_input(args, kwargs):
    if "state" in kwargs:
        return base_types.ComputationInput(
            args=args,
            kwargs=toolz.dissoc(kwargs, "state"),
            state=kwargs["state"],
        )

    return base_types.ComputationInput(args=args, kwargs=kwargs)


_incoming_edge_options = toolz.compose_left(
    graph.get_incoming_edges_for_node,
    curried.groupby(lambda edge: edge.key),
    curried.valmap(toolz.compose_left(curried.sorted(key=lambda edge: edge.priority))),
    dict.values,
    gamla.star(itertools.product),
    curried.map(tuple),
)


def _get_args(
    edges: base_types.GraphType,
    unbound_signature: base_types.NodeSignature,
    bound_signature: base_types.NodeSignature,
    results: Tuple[Tuple[base_types.ComputationResult, ...], ...],
    unbound_input: base_types.ComputationInput,
) -> Tuple[base_types.ComputationResult, ...]:
    if unbound_signature.is_args:
        return unbound_input.args

    if bound_signature.is_args:
        return toolz.pipe(
            zip(results, edges),
            curried.filter(toolz.compose_left(toolz.second, lambda edge: edge.args)),
            toolz.first,
            toolz.first,
            curried.map(_result_from_computation_result),
            tuple,
        )

    return ()


def _get_unary_computation_input(
    node: base_types.ComputationNode,
    value: base_types.ComputationResult,
    unbound_signature: base_types.NodeSignature,
) -> Dict[Text, Any]:
    return toolz.pipe(
        unbound_signature.kwargs,
        curried.filter(
            lambda arg: arg not in unbound_signature.optional_kwargs and arg != "state",
        ),
        tuple,
        gamla.check(
            gamla.anyjuxt(gamla.len_equals(1), gamla.len_equals(0)),
            _ComputationGraphException(
                "got a single input function with more than 1 unbound arguments. cannot bind function",
            ),
        ),
        gamla.curried_ternary(
            gamla.len_equals(1),
            toolz.identity,
            lambda _: node.signature.kwargs,
        ),
        lambda kwargs: {kwargs[0]: value.result},
    )


def _get_kwargs(
    edges: base_types.GraphType,
    unbound_signature: base_types.NodeSignature,
    results: Tuple[Tuple[base_types.ComputationResult, ...], ...],
    unbound_input: base_types.ComputationInput,
) -> Dict[Text, Any]:
    kwargs = toolz.pipe(
        unbound_signature.kwargs,
        curried.filter(lambda arg: arg != "state"),
        curried.map(gamla.pair_right(unbound_input.kwargs.get)),
        curried.filter(gamla.star(lambda _, value: value is not None)),
        dict,
    )

    return toolz.pipe(
        zip(edges, results),
        curried.filter(toolz.compose_left(toolz.first, lambda edge: edge.key)),
        curried.groupby(toolz.compose_left(toolz.first, lambda edge: edge.key)),
        curried.valmap(
            toolz.compose_left(
                toolz.first,
                toolz.second,
                toolz.first,
                _result_from_computation_result,
            ),
        ),
        lambda x: toolz.merge(kwargs, x),
    )


_DecisionsType = Dict[base_types.ComputationNode, base_types.ComputationResult]
_ResultDecisionPairAndNode = Tuple[
    Tuple[base_types.ComputationResult, _DecisionsType],
    base_types.ComputationNode,
]
_ResultToDecisionsType = Dict[base_types.ComputationResult, _DecisionsType]
_ResultDependenciesType = Dict[base_types.ComputationNode, _ResultToDecisionsType]


_ResultDecisionPairAndNodeTupleType = Tuple[_ResultDecisionPairAndNode, ...]


class _NotCoherent(Exception):
    """This exception signals that for a specific set of incoming
    node edges not all paths agree on the ComputationResult"""

    pass


class ComputationFailed(Exception):
    pass


_merge_decision = curried.merge_with(
    toolz.compose_left(
        toolz.unique,
        tuple,
        gamla.check(gamla.len_equals(1), _NotCoherent),
        toolz.first,
    ),
)


@toolz.curry
def _node_to_value_choices(
    result_dependencies: _ResultDependenciesType,
    node: base_types.ComputationNode,
) -> Callable[[base_types.ComputationNode], _ResultDecisionPairAndNodeTupleType]:
    return toolz.pipe(
        node,
        curried.excepts(
            KeyError,
            toolz.compose_left(result_dependencies.__getitem__, dict.items, tuple),
            gamla.ignore_input(dict),
        ),
        curried.map(lambda x: (x, node)),
        tuple,
    )


def _edge_to_value_choices(
    result_dependencies: _ResultDependenciesType,
) -> Callable[
    [base_types.ComputationEdge],
    Tuple[_ResultDecisionPairAndNodeTupleType, ...],
]:
    return toolz.compose_left(
        lambda edge: edge.args or (edge.source,),
        curried.map(_node_to_value_choices(result_dependencies)),
        gamla.star(itertools.product),
        tuple,
    )


@gamla.curry
def _edges_to_value_choices(
    result_dependencies: _ResultDependenciesType,
    edges: base_types.GraphType,
) -> Tuple[Tuple[_ResultDecisionPairAndNodeTupleType, ...], ...]:
    return toolz.pipe(
        edges,
        curried.map(_edge_to_value_choices(result_dependencies)),
        gamla.star(itertools.product),
        tuple,
    )


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
    graph,
    unbound_input,
    node: base_types.ComputationNode,
    edges: base_types.GraphType,
    results: Tuple[Tuple[base_types.ComputationResult, ...], ...],
) -> base_types.ComputationInput:
    bound_signature = base_types.NodeSignature(
        is_args=node.signature.is_args and any(edge.args for edge in edges),
        kwargs=tuple(filter(None, map(lambda edge: edge.key, edges))),
    )
    unbound_signature = _signature_difference(node.signature, bound_signature)
    unbound_input = _get_node_unbound_input(graph, unbound_input, node)

    if (
        not (unbound_signature.is_args or bound_signature.is_args)
        and sum(map(lambda edge: edge.key is None, edges)) == 1
    ):
        return base_types.ComputationInput(
            args=(),
            kwargs=_get_unary_computation_input(
                node,
                toolz.first(toolz.first(results)),
                unbound_signature,
            ),
            state=unbound_input.state,
        )

    return base_types.ComputationInput(
        args=_get_args(
            edges,
            unbound_signature,
            bound_signature,
            results,
            unbound_input,
        ),
        kwargs=_get_kwargs(edges, unbound_signature, results, unbound_input),
        state=unbound_input.state,
    )


def _wrap_in_result_if_needed(result):
    if isinstance(result, base_types.ComputationResult):
        return result
    return base_types.ComputationResult(result=result, state=None)


def to_callable(
    edges: base_types.GraphType,
    handled_exceptions: FrozenSet[Type[Exception]],
) -> Callable:
    def inner(*args, **kwargs) -> base_types.ComputationResult:
        return execute_graph(edges, handled_exceptions, args, kwargs)

    return inner


def _get_unbound_input(args, kwargs) -> base_types.ComputationInput:
    unbound_input = _make_computation_input(args, kwargs)

    if unbound_input.state is not None:
        unbound_input = dataclasses.replace(
            unbound_input,
            state=dict(unbound_input.state),
        )

    return unbound_input


@gamla.curry
def _get_node_unbound_input(
    edges: base_types.GraphType,
    unbound_input: base_types.ComputationInput,
    node: base_types.ComputationNode,
) -> base_types.ComputationInput:
    if unbound_input.state is None:
        return unbound_input
    node_id = graph.infer_node_id(edges, node)
    return dataclasses.replace(
        unbound_input,
        state=unbound_input.state[node_id] if node_id in unbound_input.state else None,
    )


def _decisions_from_value_choices(
    choices: Tuple[_ResultDecisionPairAndNodeTupleType, ...],
) -> _DecisionsType:
    return toolz.merge(
        toolz.pipe(
            choices,
            curried.concat,
            curried.map(toolz.compose_left(toolz.first, toolz.second)),
            tuple,
            gamla.curried_ternary(
                toolz.identity,
                curried.reduce(_merge_decision),
                dict,
            ),
        ),
        toolz.pipe(
            choices,
            curried.concat,
            curried.map(
                gamla.juxt(toolz.second, toolz.compose_left(toolz.first, toolz.first)),
            ),
            dict,
        ),
    )


def _choices_to_values(
    choices: Tuple[_ResultDecisionPairAndNodeTupleType, ...],
) -> Tuple[Tuple[base_types.ComputationResult, ...], ...]:
    return toolz.pipe(
        choices,
        curried.map(
            toolz.compose_left(
                curried.map(toolz.compose_left(toolz.first, toolz.first)),
                tuple,
            ),
        ),
        tuple,
    )


@toolz.curry
def _construct_computation_state(
    sink_node: base_types.ComputationNode,
    edges: base_types.GraphType,
    previous_state: Optional[Dict],
    result_dependencies: _ResultDependenciesType,
) -> Tuple[int, Any]:
    return toolz.pipe(
        toolz.merge(
            previous_state or {},
            toolz.pipe(
                toolz.merge(
                    {sink_node: toolz.first(result_dependencies[sink_node]).state},
                    toolz.pipe(
                        result_dependencies,
                        curried.get_in(
                            [sink_node, toolz.first(result_dependencies[sink_node])],
                        ),
                        curried.valmap(lambda x: x.state),
                    ),
                ),
                curried.keymap(graph.infer_node_id(edges)),
            ),
        ),
        # Convert to tuples (node id, state) so this would be hashable.
        dict.items,
        tuple,
    )


@gamla.curry
def _construct_computation_result(
    previous_state: Dict,
    edges: base_types.GraphType,
    result_dependencies: _ResultDependenciesType,
):
    return gamla.pipe(
        edges,
        graph.infer_graph_sink,
        debug_trace(edges, result_dependencies),
        gamla.pair_with(
            gamla.translate_exception(
                gamla.compose_left(result_dependencies.__getitem__, toolz.first),
                KeyError,
                ComputationFailed,
            ),
        ),
        gamla.check(toolz.identity, ComputationFailed),
        gamla.juxt(
            gamla.compose_left(toolz.first, _result_from_computation_result),
            gamla.compose_left(
                toolz.second,
                _construct_computation_state(
                    edges=edges,
                    previous_state=previous_state,
                    result_dependencies=result_dependencies,
                ),
            ),
        ),
        gamla.star(
            lambda result, state: base_types.ComputationResult(
                result=result,
                state=state,
            ),
        ),
    )


def debug_trace(
    edges: base_types.GraphType,
    result_dependencies: _ResultDependenciesType,
) -> Callable[[base_types.ComputationNode], Any]:
    if config.DEBUG_SAVE_COMPUTATION_TRACE:
        from computation_graph import visualization

        return curried.do(
            gamla.compose_left(
                node_computation_trace(result_dependencies),
                gamla.pair_with(gamla.just(edges)),
                visualization.serialize_computation_trace(
                    COMPUTATION_TRACE_DOT_FILENAME,
                ),
            ),
        )

    return toolz.identity


def node_computation_trace(
    result_dependencies: _ResultDependenciesType,
) -> Callable[
    [base_types.ComputationNode],
    Tuple[Tuple[base_types.ComputationNode, base_types.ComputationResult]],
]:
    return gamla.compose_left(
        gamla.juxt(
            gamla.pair_right(
                toolz.compose_left(
                    lambda sink: result_dependencies[sink],
                    dict.keys,
                    toolz.first,
                ),
            ),
            gamla.compose_left(
                lambda sink: result_dependencies[sink],
                dict.values,
                toolz.first,
                dict.items,
            ),
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
def _run_keeping_choices(
    make_input,
    node,
    incoming_edges,
    value_choices,
):
    node_input = make_input(
        node,
        incoming_edges,
        _choices_to_values(value_choices),
    )
    return (
        _wrap_in_result_if_needed(_apply(node, node_input)),
        _decisions_from_value_choices(value_choices),
    )


@gamla.curry
def _process_layer_in_parallel(f, state, layer):
    return toolz.merge(state, *gamla.map(f(state), layer))


@gamla.curry
def _dag_reduce(reducer: Callable, graph: base_types.GraphType):
    """Directed acyclic graph reduction."""
    return gamla.pipe(
        graph,
        _toposort_nodes,
        gamla.reduce(
            _process_layer_in_parallel(reducer),
            {},
        ),
    )


@gamla.curry
def _per_values_option(excepting, f, accumulated_outputs, node, edge_option):
    return gamla.pipe(
        edge_option,
        gamla.juxt(gamla.wrap_tuple, _edges_to_value_choices(accumulated_outputs)),
        gamla.star(itertools.product),
        curried.map(gamla.star(excepting(f(node)))),
        curried.filter(None),
        dict,
    )


@gamla.curry
def _per_edge_option(f, edge_options, accumulated_outputs, node):
    return gamla.pipe(
        node,
        edge_options,
        curried.map(f(accumulated_outputs, node)),
        gamla.star(toolz.merge),
        curried.assoc(accumulated_outputs, node),
    )


def execute_graph(
    graph: base_types.GraphType,
    handled_exceptions: FrozenSet[Type[Exception]],
    args,
    kwargs,
) -> base_types.ComputationResult:
    unbound_input = _get_unbound_input(args, kwargs)
    return gamla.pipe(
        graph,
        _dag_reduce(
            _per_edge_option(
                _per_values_option(
                    lambda f: toolz.excepts(
                        tuple(handled_exceptions) + (_NotCoherent,),
                        f,
                        gamla.compose_left(
                            type,
                            _log_handled_exception,
                            gamla.just(None),
                        ),
                    ),
                    _run_keeping_choices(
                        _get_computation_input(graph, unbound_input),
                    ),
                ),
                lambda node: _incoming_edge_options(graph, node),
            ),
        ),
        _construct_computation_result(unbound_input.state, graph),
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
