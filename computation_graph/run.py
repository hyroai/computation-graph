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
from toolz.curried import operator

from computation_graph import base_types, graph


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


_get_node_ambiguous_edge_groups = toolz.compose_left(
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
            tuple,
        )

    return ()


def _get_unary_computation_input(
    node: base_types.ComputationNode,
    value: Any,
    unbound_signature: base_types.NodeSignature,
) -> Dict[Text, Any]:
    return toolz.pipe(
        unbound_signature.kwargs,
        curried.filter(lambda arg: arg not in unbound_signature.optional_kwargs),
        tuple,
        gamla.check(
            gamla.anyjuxt(gamla.len_equals(1), gamla.len_equals(0)),
            _ComputationGraphException(
                "got a unary function with more than 1 unbound arguments. cannot bind function",
            ),
        ),
        gamla.curried_ternary(
            gamla.len_equals(1), toolz.identity, lambda _: node.signature.kwargs,
        ),
        lambda kwargs: {kwargs[0]: value},
    )


def _get_kwargs(
    edges: base_types.GraphType,
    unbound_signature: base_types.NodeSignature,
    results: Tuple[Tuple[base_types.ComputationResult, ...], ...],
    unbound_input: base_types.ComputationInput,
) -> Dict[Text, Any]:
    kwargs = toolz.pipe(
        unbound_signature.kwargs,
        curried.map(gamla.pair_right(unbound_input.kwargs.get)),
        curried.filter(gamla.star(lambda _, value: value is not None)),
        dict,
    )

    return toolz.pipe(
        zip(edges, results),
        curried.filter(toolz.compose_left(toolz.first, lambda edge: edge.key)),
        curried.groupby(toolz.compose_left(toolz.first, lambda edge: edge.key)),
        curried.valmap(toolz.compose_left(toolz.first, toolz.second, toolz.first)),
        lambda x: toolz.merge(kwargs, x),
    )


_DecisionsType = Dict[base_types.ComputationNode, base_types.ComputationResult]
_ResultDecisionPairAndNode = Tuple[
    Tuple[base_types.ComputationResult, _DecisionsType], base_types.ComputationNode,
]
_ResultToDecisionsType = Dict[base_types.ComputationResult, _DecisionsType]
_ResultDependenciesType = Dict[base_types.ComputationNode, _ResultToDecisionsType]

_ComputationResultAndNodeType = Tuple[
    base_types.ComputationResult, base_types.ComputationNode,
]

_ResultDecisionPairAndNodeTupleType = Tuple[_ResultDecisionPairAndNode, ...]


class _NotCoherent(Exception):
    """This exception signals that for a specific set of incoming
    node edges not all paths agree on the ComputationResult"""

    pass


class _ComputationFailed(Exception):
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
    result_dependencies: _ResultDependenciesType, node: base_types.ComputationNode,
) -> _ResultDecisionPairAndNodeTupleType:
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
    [base_types.ComputationEdge], Tuple[_ResultDecisionPairAndNodeTupleType, ...],
]:
    return toolz.compose_left(
        lambda edge: edge.args or (edge.source,),
        curried.map(_node_to_value_choices(result_dependencies)),
        gamla.star(itertools.product),
        tuple,
    )


@dataclasses.dataclass(frozen=True)
class _PastNode:
    node: base_types.ComputationNode


def _future_edge_to_value_choices(
    previous_results: _DecisionsType,
) -> Callable[
    [base_types.ComputationEdge], Tuple[_ResultDecisionPairAndNodeTupleType, ...],
]:
    return toolz.compose_left(
        lambda edge: edge.source,
        gamla.ternary(
            operator.contains(previous_results),
            gamla.pair_with(previous_results.__getitem__),
            gamla.juxt(gamla.just(None), _PastNode),
        ),
        gamla.star(
            lambda result, node: ((result, {_PastNode(node): result}), _PastNode(node)),
        ),
        gamla.wrap_tuple,
        gamla.wrap_tuple,
    )


def _edges_to_value_choices(
    edges: base_types.GraphType,
    result_dependencies: _ResultDependenciesType,
    previous_results: _DecisionsType,
) -> Tuple[Tuple[_ResultDecisionPairAndNodeTupleType, ...], ...]:
    return toolz.pipe(
        edges,
        curried.map(
            gamla.ternary(
                lambda edge: edge.is_future,
                _future_edge_to_value_choices(previous_results),
                _edge_to_value_choices(result_dependencies),
            ),
        ),
        gamla.star(itertools.product),
        tuple,
    )


def _signature_difference(
    sig_a: base_types.NodeSignature, sig_b: base_types.NodeSignature,
) -> base_types.NodeSignature:
    return base_types.NodeSignature(
        is_args=(sig_a.is_args != sig_b.is_args),
        # Difference must save the order of the left signature.
        kwargs=tuple(filter(lambda x: x not in sig_b.kwargs, sig_a.kwargs)),
        optional_kwargs=tuple(
            filter(lambda x: x not in sig_b.optional_kwargs, sig_a.optional_kwargs),
        ),
    )


def _get_computation_input(
    edges: base_types.GraphType,
    node: base_types.ComputationNode,
    results: Tuple[Tuple[base_types.ComputationResult, ...], ...],
    unbound_input: base_types.ComputationInput,
) -> base_types.ComputationInput:
    bound_signature = base_types.NodeSignature(
        is_args=node.signature.is_args and any(edge.args for edge in edges),
        kwargs=tuple(filter(None, map(lambda edge: edge.key, edges))),
    )
    unbound_signature = _signature_difference(node.signature, bound_signature)

    if (
        not (unbound_signature.is_args or bound_signature.is_args)
        and sum(map(lambda edge: edge.key is None, edges)) == 1
    ):
        return base_types.ComputationInput(
            args=(),
            kwargs=_get_unary_computation_input(
                node, toolz.first(toolz.first(results)), unbound_signature,
            ),
        )

    return base_types.ComputationInput(
        args=_get_args(
            edges, unbound_signature, bound_signature, results, unbound_input,
        ),
        kwargs=_get_kwargs(edges, unbound_signature, results, unbound_input),
    )


def _apply(
    node: base_types.ComputationNode, node_input: base_types.ComputationInput,
) -> base_types.ComputationResult:
    assert node.func is not None, f"cannot apply {node}"
    return node.func(*node_input.args, **node_input.kwargs)


def _edge_with_values_to_computation_result_and_node(
    edge: base_types.ComputationEdge, values: Tuple[base_types.ComputationResult, ...],
) -> Tuple[_ComputationResultAndNodeType, ...]:
    return toolz.pipe(zip(edge.args or (edge.source,), values), tuple)


def to_callable(
    edges: base_types.GraphType, handled_exceptions: FrozenSet[Type[Exception]],
) -> Callable:
    def inner(*args, **kwargs) -> base_types.ComputationResult:
        return execute_graph(
            edges,
            handled_exceptions,
            args,
            kwargs,
            kwargs["state"] if "state" in kwargs else (),
        )

    return inner


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
                toolz.identity, curried.reduce(_merge_decision), dict,
            ),
        ),
        toolz.pipe(
            choices,
            curried.concat,
            curried.map(
                toolz.juxt(toolz.second, toolz.compose_left(toolz.first, toolz.first)),
            ),
            dict,
        ),
    )


def _results_from_value_choices(
    choices: Tuple[_ResultDecisionPairAndNodeTupleType, ...],
) -> Tuple[Tuple[base_types.ComputationResult, ...], ...]:
    return toolz.pipe(
        choices,
        curried.map(
            toolz.compose_left(
                curried.map(toolz.compose_left(toolz.first, toolz.first)), tuple,
            ),
        ),
        tuple,
    )


def _construct_computation_result(
    edges: base_types.GraphType,
    result_dependencies: _ResultDependenciesType,
    previous_state: _DecisionsType,
):
    return toolz.pipe(
        edges,
        graph.infer_graph_sink,
        gamla.pair_with(
            gamla.translate_exception(
                lambda sink: toolz.first(result_dependencies[sink]),
                KeyError,
                _ComputationFailed,
            ),
        ),
        gamla.check(toolz.identity, _ComputationFailed),
        curried.juxt(
            toolz.first,
            toolz.compose_left(
                toolz.second,
                curried.juxt(
                    toolz.compose_left(
                        result_dependencies.get, dict.items, toolz.first, toolz.second,
                    ),
                    lambda sink: {sink: toolz.first(result_dependencies[sink])},
                ),
                gamla.star(curried.merge),
                curried.keyfilter(lambda node: not isinstance(node, _PastNode)),
                curried.keymap(graph.infer_node_id(edges)),
                lambda state: toolz.merge(previous_state, state),
                # Convert to tuples (node id, state) so this would be hashable.
                dict.items,
                tuple,
            ),
        ),
        gamla.star(
            lambda result, state: base_types.ComputationResult(
                result=result, state=state,
            ),
        ),
    )


_filter_future_edges = toolz.compose_left(
    curried.filter(lambda edge: not edge.is_future), tuple,
)


def execute_graph(
    edges: base_types.GraphType,
    handled_exceptions: FrozenSet[Type[Exception]],
    args,
    kwargs,
    state: Tuple,
) -> base_types.ComputationResult:
    unbound_input = base_types.ComputationInput(args=args, kwargs=kwargs)

    result_dependencies: _ResultDependenciesType = {}
    prev_results = toolz.pipe(
        state, dict, curried.keymap(graph.infer_node_from_id(edges)),
    )
    for node_set in toolz.pipe(edges, _filter_future_edges, _toposort_nodes):
        for node in node_set:
            for node_edges in _get_node_ambiguous_edge_groups(edges, node):
                for choices in _edges_to_value_choices(
                    node_edges, result_dependencies, prev_results,
                ):
                    try:
                        decisions = _decisions_from_value_choices(choices)
                    except _NotCoherent:
                        continue

                    computation_input = _get_computation_input(
                        node_edges,
                        node,
                        _results_from_value_choices(choices),
                        unbound_input,
                    )

                    try:
                        if node not in result_dependencies:
                            result_dependencies[node] = {}

                        result_dependencies[node][
                            _apply(node, computation_input)
                        ] = decisions

                    except tuple(handled_exceptions) as exception:
                        _log_handled_exception(type(exception))

    try:
        return _construct_computation_result(edges, result_dependencies, dict(state))
    except _ComputationFailed:
        raise base_types.ExhaustedAllComputationPaths()


def _log_handled_exception(exception_type: Type[Exception]):
    _, exception, exception_traceback = sys.exc_info()
    filename, line_num, func_name, _ = traceback.extract_tb(exception_traceback)[-1]
    if str(exception):
        reason = f": {exception}"
    else:
        reason = ""
    code_location = f"{pathlib.Path(filename).name}:{line_num}"
    logging.debug(f"'{func_name.strip('_')}' {exception_type}@{code_location}{reason}")
