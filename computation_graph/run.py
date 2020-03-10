import collections
import dataclasses
import itertools
import logging
import pathlib
import sys
import traceback
from typing import (
    Any,
    Callable,
    DefaultDict,
    Dict,
    FrozenSet,
    Optional,
    Set,
    Text,
    Tuple,
    Type,
)

import gamla
import toolz
import toposort
from toolz import curried

from computation_graph import base_types, graph


class _ComputationGraphException(Exception):
    pass


def _transpose_graph(
    graph: Dict[base_types.ComputationNode, Set[base_types.ComputationNode]]
) -> DefaultDict:
    transposed_graph: DefaultDict = collections.defaultdict(set)
    for source, destination_set in graph.items():
        for node in destination_set:
            transposed_graph[node].add(source)

    return transposed_graph


def _toposort_nodes(
    edges: base_types.GraphType,
) -> Tuple[FrozenSet[base_types.ComputationNode], ...]:
    dependencies: DefaultDict = collections.defaultdict(set)
    for edge in edges:
        if edge.args:
            for arg in edge.args:
                dependencies[arg].add(edge.destination)
        else:
            dependencies[edge.source].add(edge.destination)

    return tuple(map(frozenset, toposort.toposort(_transpose_graph(dependencies))))


def _make_computation_input(args, kwargs):
    if "state" in kwargs:
        return base_types.ComputationInput(
            args=args, kwargs=toolz.dissoc(kwargs, "state"), state=kwargs["state"]
        )

    return base_types.ComputationInput(args=args, kwargs=kwargs)


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
            curried.map(lambda computation_result: computation_result.result),
            tuple,
        )

    return ()


def _get_unary_computation_input(
    node: base_types.ComputationNode,
    value: base_types.ComputationResult,
    unbound_signature: base_types.NodeSignature,
    unbound_input: base_types.ComputationInput,
) -> base_types.ComputationInput:
    unbound_kwargs = tuple(
        filter(
            lambda x: x not in unbound_signature.optional_kwargs and x != "state",
            unbound_signature.kwargs,
        )
    )
    if len(unbound_kwargs) == 1:
        kwargs = unbound_kwargs
    elif len(unbound_kwargs) == 0:
        kwargs = node.signature.kwargs
    else:
        raise _ComputationGraphException(
            "got a single input function with more than 1 unbound arguments. cannot bind function"
        )

    return base_types.ComputationInput(
        args=(), kwargs={kwargs[0]: value.result}, state=unbound_input.state
    )


def _get_kwargs(
    edges: base_types.GraphType,
    node: base_types.ComputationNode,
    unbound_signature: base_types.NodeSignature,
    results: Tuple[Tuple[base_types.ComputationResult, ...], ...],
    unbound_input: base_types.ComputationInput,
) -> Dict[Text, Any]:
    kwargs = {}

    for key in unbound_signature.kwargs:
        if key == "state":
            continue
        try:
            kwargs[key] = unbound_input.kwargs[key]
        except KeyError:
            if key in unbound_signature.optional_kwargs:
                continue

            raise _ComputationGraphException(
                f"could not bind parameter {key} for node {node}"
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
                lambda computation_result: computation_result.result,
            )
        ),
        lambda x: toolz.merge(kwargs, x),
    )


_DecisionsType = Dict[base_types.ComputationNode, base_types.ComputationResult]
_ResultDecisionPairAndNode = Tuple[
    Tuple[base_types.ComputationResult, _DecisionsType], base_types.ComputationNode
]
_ResultToDecisionsType = Dict[base_types.ComputationResult, _DecisionsType]
_ResultDependenciesType = Dict[base_types.ComputationNode, _ResultToDecisionsType]

_ComputationResultAndNodeType = Tuple[
    base_types.ComputationResult, base_types.ComputationNode
]


class _NotCoherent(Exception):
    pass


class _ComputationFailed(Exception):
    pass


_merge_decision = curried.merge_with(
    toolz.compose_left(
        toolz.unique,
        tuple,
        gamla.check(lambda x: len(x) == 1, _NotCoherent),
        toolz.first,
    )
)


@toolz.curry
def _node_to_value_choices(
    result_dependencies: _ResultDependenciesType, node: base_types.ComputationNode
) -> Callable[[base_types.ComputationNode], Tuple[_ResultDecisionPairAndNode, ...]]:
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
    [base_types.ComputationEdge], Tuple[Tuple[_ResultDecisionPairAndNode, ...], ...]
]:
    return toolz.compose_left(
        lambda edge: edge.args or (edge.source,),
        curried.map(_node_to_value_choices(result_dependencies)),
        gamla.star(itertools.product),
        tuple,
    )


def _edges_to_value_choices(
    edges: base_types.GraphType, result_dependencies: _ResultDependenciesType
) -> Tuple[Tuple[Tuple[_ResultDecisionPairAndNode, ...], ...], ...]:
    return toolz.pipe(
        edges,
        curried.map(_edge_to_value_choices(result_dependencies)),
        gamla.star(itertools.product),
        tuple,
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
        return _get_unary_computation_input(
            node, toolz.first(toolz.first(results)), unbound_signature, unbound_input
        )

    return base_types.ComputationInput(
        args=_get_args(
            edges, unbound_signature, bound_signature, results, unbound_input
        ),
        kwargs=_get_kwargs(edges, node, unbound_signature, results, unbound_input),
        state=unbound_input.state,
    )


def _apply(
    node: base_types.ComputationNode, node_input: base_types.ComputationInput
) -> base_types.ComputationResult:
    assert node.func is not None, f"cannot apply {node}"
    if node.is_stateful:
        return node.func(
            *node_input.args,
            **toolz.assoc(node_input.kwargs, "state", node_input.state),
        )

    result = node.func(*node_input.args, **node_input.kwargs)

    # We could have gotten a `ComputationResult` (In the case `node` is a nested `ComputationNode` for example).
    # Unwrap to avoid nested results.
    if isinstance(result, base_types.ComputationResult):
        return base_types.ComputationResult(result.result, result.state)

    return base_types.ComputationResult(result=result, state=node_input.state)


def _edge_with_values_to_computation_result_and_node(
    edge: base_types.ComputationEdge, values: Tuple[base_types.ComputationResult, ...]
) -> Tuple[_ComputationResultAndNodeType, ...]:
    return toolz.pipe(zip(edge.args or (edge.source,), values), tuple)


def to_callable(edges: base_types.GraphType) -> Callable:
    def inner(*args, **kwargs) -> base_types.ComputationResult:
        return execute_graph(edges, args, kwargs)

    return inner


def _get_unbound_input(args, kwargs) -> base_types.ComputationInput:
    unbound_input = _make_computation_input(args, kwargs)

    if unbound_input.state is not None:
        unbound_input = dataclasses.replace(
            unbound_input, state=dict(unbound_input.state)
        )

    return unbound_input


def _get_node_unbound_input(
    edges: base_types.GraphType,
    node: base_types.ComputationNode,
    unbound_input: base_types.ComputationInput,
) -> base_types.ComputationInput:
    if unbound_input.state is None:
        return unbound_input
    return dataclasses.replace(
        unbound_input,
        state=unbound_input.state[graph.infer_node_id(edges, node)]
        if graph.infer_node_id(edges, node) in unbound_input.state
        else None,
    )


def _get_allowed_exceptions_for_node(
    edges: base_types.GraphType, node: base_types.ComputationNode
) -> Tuple[Type[Exception], ...]:
    return toolz.pipe(
        graph.get_outgoing_edges_for_node(edges, node),
        curried.mapcat(lambda edge: edge.allowed_exceptions),
        curried.unique,
        tuple,
    )


def _decisions_from_value_choices(
    choices: Tuple[Tuple[_ResultDecisionPairAndNode, ...], ...]
) -> _DecisionsType:
    return toolz.merge(
        toolz.pipe(
            choices,
            curried.concat,
            tuple,
            curried.map(toolz.compose_left(toolz.first, toolz.second)),
            tuple,
            gamla.curried_ternary(
                toolz.identity, curried.reduce(_merge_decision), dict
            ),
        ),
        toolz.pipe(
            choices,
            curried.concat,
            curried.map(
                toolz.juxt(toolz.second, toolz.compose_left(toolz.first, toolz.first),)
            ),
            dict,
        ),
    )


def _results_from_value_choices(
    choices: Tuple[Tuple[_ResultDecisionPairAndNode, ...], ...]
) -> Tuple[Tuple[base_types.ComputationResult, ...], ...]:
    if choices:
        return toolz.pipe(
            choices,
            curried.map(
                toolz.compose_left(
                    curried.map(toolz.compose_left(toolz.first, toolz.first)), tuple,
                )
            ),
            tuple,
        )

    return ()


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
                            [sink_node, toolz.first(result_dependencies[sink_node])]
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


def _construct_computation_result(
    edges: base_types.GraphType,
    result_dependencies: _ResultDependenciesType,
    previous_state: Dict,
):
    return toolz.pipe(
        edges,
        graph.infer_graph_sink,
        gamla.pair_with(
            gamla.translate_exception(
                lambda sink: toolz.first(result_dependencies[sink]),
                KeyError,
                _ComputationFailed,
            )
        ),
        curried.juxt(
            toolz.compose_left(
                toolz.first, lambda computation_result: computation_result.result
            ),
            toolz.compose_left(
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
                result=result, state=state
            )
        ),
    )


def execute_graph(
    edges: base_types.GraphType, args, kwargs
) -> base_types.ComputationResult:
    unbound_input = _get_unbound_input(args, kwargs)
    last_exception: Exception = StopIteration()
    result_dependencies: _ResultDependenciesType = {}
    for node_set in _toposort_nodes(edges):
        for node in node_set:
            node_unbound_input = _get_node_unbound_input(edges, node, unbound_input)
            for node_edges in _get_node_ambiguous_edge_groups(edges, node):
                exceptions = _get_allowed_exceptions_for_node(edges, node)
                for choices in _edges_to_value_choices(node_edges, result_dependencies):
                    try:
                        decisions = _decisions_from_value_choices(choices)
                    except _NotCoherent:
                        continue

                    computation_input = _get_computation_input(
                        node_edges,
                        node,
                        _results_from_value_choices(choices),
                        node_unbound_input,
                    )

                    try:
                        if node not in result_dependencies:
                            result_dependencies[node] = {}

                        result_dependencies[node][
                            _apply(node, computation_input)
                        ] = decisions

                    except exceptions as exception:
                        _log_unactionable_source(type(exception))
                        last_exception = exception
    try:
        return _construct_computation_result(
            edges, result_dependencies, unbound_input.state
        )
    except _ComputationFailed:
        raise last_exception


def _log_unactionable_source(exception_type: Type[Exception]):
    """Must be run from within an `except Unactionable` clause. """
    _, exception, exception_traceback = sys.exc_info()
    filename, line_num, func_name, _ = traceback.extract_tb(exception_traceback)[-1]
    if str(exception):
        reason = f": {exception}"
    else:
        reason = ""
    code_location = f"{pathlib.Path(filename).name}:{line_num}"
    logging.debug(f"'{func_name.strip('_')}' {exception_type}@{code_location}{reason}")
