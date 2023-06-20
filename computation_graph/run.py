import asyncio
import dataclasses
import functools
import itertools
import logging
import time
import typing
from typing import Any, Callable, Dict, FrozenSet, Set, Tuple, Type

import gamla
import immutables
import termcolor
import toposort
import typeguard
from gamla.optimized import async_functions as opt_async_gamla
from gamla.optimized import sync as opt_gamla

from computation_graph import base_types, graph, signature
from computation_graph.composers import debug


class _DepNotFoundError(Exception):
    pass


_NodeToResults = Dict[base_types.ComputationNode, base_types.Result]
_ComputationInput = Tuple[Tuple[base_types.Result, ...], Dict[str, base_types.Result]]
_SingleNodeSideEffect = Callable[[base_types.ComputationNode, Any], None]
_NodeToComputationInput = Callable[[base_types.ComputationNode], _ComputationInput]
_ComputaionInputSpec = Tuple[
    Tuple[base_types.ComputationNode, ...], Dict[str, base_types.ComputationNode]
]


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


def _type_check(node: base_types.ComputationNode, result):
    return_typing = typing.get_type_hints(node.func).get("return", None)
    if return_typing:
        try:
            typeguard.check_type(str(node), result, return_typing)
        except TypeError as e:
            logging.error([node.func.__code__, e])


def _profile(node, time_started: float):
    elapsed = time.perf_counter() - time_started
    if elapsed <= 0.1:
        return
    logging.warning(
        termcolor.colored(
            f"function took {elapsed:.6f} seconds: {base_types.pretty_print_function_name(node.func)}",
            color="red",
        )
    )


def _make_get_node_input_and_apply(
    is_async: bool, side_effect: _SingleNodeSideEffect
) -> Callable[..., base_types.Result]:

    if is_async:

        @opt_async_gamla.star
        async def run_node(
            node_to_input: _NodeToComputationInput, node: base_types.ComputationNode
        ) -> base_types.Result:
            args, kwargs = node_to_input(node)
            before = time.perf_counter()
            result = await gamla.to_awaitable(node.func(*args, **kwargs))
            side_effect(node, result)
            _profile(node, before)
            return node, result

    else:

        @opt_gamla.star
        def run_node(node_to_input: _NodeToComputationInput, node: base_types.ComputationNode) -> base_types.Result:  # type: ignore
            args, kwargs = node_to_input(node)
            before = time.perf_counter()
            result = node.func(*args, **kwargs)
            side_effect(node, result)
            _profile(node, before)
            return node, result

    return run_node


_is_graph_async = opt_gamla.compose_left(
    opt_gamla.mapcat(lambda edge: (edge.source, *edge.args)),
    opt_gamla.remove(gamla.equals(None)),
    opt_gamla.map(base_types.node_implementation),
    gamla.anymap(asyncio.iscoroutinefunction),
)
_assert_no_unwanted_ambiguity = gamla.compose_left(
    base_types.ambiguity_groups,
    gamla.assert_that_with_message(
        gamla.wrap_str(
            "There are multiple edges with the same destination, key and priority in the computation graph!: {}"
        ),
        gamla.len_equals(0),
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


_get_args_nodes: Callable[
    [Tuple[base_types.ComputationEdge, ...]], Tuple[base_types.ComputationNode, ...],
] = gamla.compose_left(
    opt_gamla.filter(base_types.edge_args), gamla.head, base_types.edge_args
)
_get_kwargs_nodes = opt_gamla.compose_left(
    opt_gamla.filter(
        gamla.compose_left(base_types.edge_key, gamla.not_equals("*args"))
    ),
    gamla.map(gamla.juxt(base_types.edge_key, base_types.edge_source)),
    gamla.frozendict,
)


def _node_incoming_edges_to_input_spec(
    node_incoming_edges: Tuple[base_types.ComputationEdge],
) -> _ComputaionInputSpec:
    if not len(node_incoming_edges):
        return (), {}
    first_incoming_edge = gamla.head(node_incoming_edges)
    node = base_types.edge_destination(first_incoming_edge)
    if node.signature.is_kwargs:
        return (base_types.edge_source(first_incoming_edge),), {}
    return (
        _get_args_nodes(node_incoming_edges) if node.signature.is_args else (),
        _get_kwargs_nodes(node_incoming_edges),
    )


def _edges_to_accumulated_results_to_node_to_first_possible_input(
    edges,
) -> Callable[[_NodeToResults], _NodeToComputationInput,]:
    node_to_incoming_edges = graph.get_incoming_edges_for_node(edges)
    node_to_computation_input_spec_options: Callable[
        [base_types.ComputationNode], Tuple[_ComputaionInputSpec]
    ] = functools.cache(
        gamla.compose_left(
            node_to_incoming_edges,
            opt_gamla.groupby(base_types.edge_key),
            opt_gamla.valmap(gamla.sort_by(base_types.edge_priority)),
            dict.values,
            opt_gamla.star(itertools.product),
            opt_gamla.maptuple(_node_incoming_edges_to_input_spec),
        )
    )
    # TODO(eli): Add sync.translate_exception/except that avoids inspect.iscoroutinefunction so this can be inlined (ENG-5212)
    head_or_dep_not_found = gamla.translate_exception(
        gamla.head, StopIteration, _DepNotFoundError
    )

    def make_node_to_first_possible_input(
        accumulated_results: _NodeToResults,
    ) -> _NodeToComputationInput:
        return opt_gamla.compose_left(
            node_to_computation_input_spec_options,
            opt_gamla.map(
                gamla.excepts(
                    KeyError,
                    lambda _: None,
                    opt_gamla.packstack(
                        opt_gamla.maptuple(accumulated_results.__getitem__),
                        opt_gamla.valmap(accumulated_results.__getitem__),
                    ),
                )
            ),
            opt_gamla.remove(gamla.equals(None)),
            head_or_dep_not_found,
            tuple,
        )

    return make_node_to_first_possible_input


def _make_reduce_layers(
    edges: base_types.GraphType,
    handled_exceptions: Tuple[Type[Exception], ...],
    is_async: bool,
    single_node_side_effect: _SingleNodeSideEffect,
) -> Callable[[immutables.Map], Callable[[Tuple[FrozenSet, ...]], immutables.Map]]:
    get_node_input_and_apply = _make_get_node_input_and_apply(
        is_async, single_node_side_effect
    )
    accumlated_results_to_node_to_input = (
        _edges_to_accumulated_results_to_node_to_first_possible_input(edges)
    )
    single_layer_reducer = debug.name_callable(
        gamla.compose_left(
            gamla.pack,
            gamla.juxt(
                gamla.head,
                gamla.compose_left(
                    opt_gamla.packstack(
                        accumlated_results_to_node_to_input, gamla.identity
                    ),
                    gamla.explode(1),
                    gamla.map_filter_empty(
                        gamla.first(
                            get_node_input_and_apply,
                            gamla.just(None),
                            exception_type=(
                                *handled_exceptions,
                                base_types.SkipComputationError,
                                _DepNotFoundError,
                            ),
                        )
                    ),
                    tuple,
                ),
            ),
            opt_gamla.star(
                lambda prev_results, current_results: prev_results.update(
                    current_results
                )
            ),
        ),
        "single_layer_reducer",
    )
    return gamla.curry(gamla.reduce_curried)(single_layer_reducer)


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
