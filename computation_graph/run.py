import asyncio
import dataclasses
import functools
import inspect
import itertools
import logging
import os
import time
import typing
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    Mapping,
    Optional,
    Set,
    Tuple,
    Type,
)

import gamla
import immutables
import termcolor
import toposort
import typeguard
from gamla.optimized import sync as opt_gamla

from computation_graph import base_types, composers, graph, signature

CG_NO_RESULT = "CG_NO_RESULT"


class _DepNotFoundError(Exception):
    pass


_NodeToResults = Dict[base_types.ComputationNode, base_types.Result]
_ComputationInput = Tuple[Tuple[base_types.Result, ...], Dict[str, base_types.Result]]
_SingleNodeSideEffect = Callable[[base_types.ComputationNode, Any], None]
_ComputationInputSpec = Tuple[
    Tuple[base_types.ComputationNode, ...], Dict[str, base_types.ComputationNode]
]
_NodeExecutor = Callable[
    [
        Mapping[
            base_types.ComputationNode,
            base_types.Result | Awaitable[base_types.Result],
        ],
        base_types.ComputationNode,
    ],
    base_types.Result,
]


def _transpose_graph(
    graph: Dict[base_types.ComputationNode, Set[base_types.ComputationNode]]
) -> Dict[base_types.ComputationNode, Set[base_types.ComputationNode]]:
    return opt_gamla.pipe(
        graph, dict.keys, opt_gamla.groupby_many(graph.get), opt_gamla.valmap(set)
    )


_toposort_nodes: Callable[
    [base_types.GraphType], Tuple[FrozenSet[base_types.ComputationNode], ...]
] = opt_gamla.compose_left(
    opt_gamla.groupby_many(base_types.edge_sources),
    opt_gamla.valmap(
        opt_gamla.compose_left(opt_gamla.map(base_types.edge_destination), set)
    ),
    _transpose_graph,
    lambda g: toposort.toposort_flatten(g, sort=False),
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


_group_by_is_async_result = opt_gamla.groupby(lambda k_v: inspect.isawaitable(k_v[1]))


_is_graph_async = opt_gamla.compose_left(
    opt_gamla.mapcat(lambda edge: (edge.source, *edge.args)),
    opt_gamla.remove(gamla.equals(None)),
    opt_gamla.map(base_types.node_implementation),
    gamla.anymap(asyncio.iscoroutinefunction),
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


"""Replace multiple edges pointing to a terminal with one edge that has multiple args"""


def _merge_edges_pointing_to_terminals(g: base_types.GraphType) -> base_types.GraphType:
    return gamla.compose_left(
        gamla.groupby(gamla.attrgetter("destination")),
        gamla.itemmap(
            gamla.star(
                lambda dest, edges_for_dest: (
                    dest,
                    base_types.merge_graphs(
                        composers.make_or(
                            opt_gamla.maptuple(base_types.edge_source)(edges_for_dest),
                            merge_fn=(aggregate := lambda args: args),
                        ),
                        composers.compose_left_unary(aggregate, dest),
                    )
                    if dest.is_terminal
                    else edges_for_dest,
                )
            )
        ),
        dict.values,
        gamla.concat,
        tuple,
    )(g)


def _to_callable_with_side_effect_for_single_and_multiple(
    single_node_side_effect: _SingleNodeSideEffect,
    all_nodes_side_effect: Callable,
    edges: base_types.GraphType,
    handled_exceptions: Tuple[Type[Exception], ...],
) -> Callable[[_NodeToResults, _NodeToResults], _NodeToResults]:
    edges = _merge_edges_pointing_to_terminals(edges)
    single_node_side_effect = (
        (lambda node, result: result)
        if os.getenv(base_types.COMPUTATION_GRAPH_DEBUG_ENV_KEY) is None
        else single_node_side_effect
    )

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
        gamla.side_effect(base_types.assert_no_unwanted_ambiguity),
    )
    is_async = _is_graph_async(edges)
    placeholder_to_future_source = opt_gamla.pipe(
        future_source_to_placeholder, gamla.itemmap(lambda k_v: (k_v[1], k_v[0]))
    )
    get_node_executor = _make_get_node_executor(
        edges, handled_exceptions, single_node_side_effect
    )

    topological_layers = opt_gamla.pipe(
        edges,
        _toposort_nodes,
        gamla.remove(gamla.contains(placeholder_to_future_source)),
        opt_gamla.maptuple(opt_gamla.pair_right(get_node_executor)),
    )

    translate_source_to_placeholder = opt_gamla.compose_left(
        opt_gamla.keyfilter(gamla.contains(future_sources)),
        opt_gamla.keymap(future_source_to_placeholder.__getitem__),
    )
    all_node_side_effects_on_edges = gamla.side_effect(all_nodes_side_effect(edges))

    if is_async:

        async def final_runner(sources_to_values):
            d = gamla.pipe(
                topological_layers,
                _run_graph(
                    translate_source_to_placeholder(sources_to_values),
                    handled_exceptions,
                ),
                dict,
            )
            results_by_is_async = _group_by_is_async_result(d.items())
            async_results = tuple(zip(*results_by_is_async.get(True, ())))
            sync_results = dict(results_by_is_async.get(False, ()))
            return all_node_side_effects_on_edges(
                sync_results
                | (
                    {
                        k: v
                        for (k, v) in zip(
                            async_results[0],
                            await asyncio.gather(
                                *async_results[1], return_exceptions=True
                            ),
                        )
                        if not isinstance(v, Exception)
                    }
                    if async_results
                    else {}
                )
            )

    else:

        def final_runner(sources_to_values):
            return gamla.pipe(
                topological_layers,
                _run_graph(
                    translate_source_to_placeholder(sources_to_values),
                    handled_exceptions,
                ),
                dict,
                all_node_side_effects_on_edges,
            )

    return (_async_graph_reducer if is_async else _graph_reducer)(final_runner)


_get_args_nodes: Callable[
    [Tuple[base_types.ComputationEdge, ...]], Tuple[base_types.ComputationNode, ...]
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
) -> _ComputationInputSpec:
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


def _to_awaitable(v) -> Awaitable:
    if inspect.isawaitable(v):
        return v
    f = asyncio.get_event_loop().create_future()
    f.set_result(v)
    return f


def _make_get_node_executor(
    edges, handled_exceptions, single_node_side_effect: _SingleNodeSideEffect
):
    node_to_incoming_edges = functools.cache(graph.get_incoming_edges_for_node(edges))
    node_to_computation_input_spec_options: Callable[
        [base_types.ComputationNode], Tuple[_ComputationInputSpec]
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

    async def gather(
        args: Tuple[Awaitable, ...], kwargs: Mapping[str, Awaitable]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        if args or kwargs:
            try:
                gathered = await asyncio.gather(*args, *kwargs.values())
                return gathered[: len(args)], dict(
                    zip(kwargs.keys(), gathered[len(args) :])
                )
            except (
                _DepNotFoundError,
                base_types.SkipComputationError,
                *handled_exceptions,
            ):
                # We delete the references to the upstream tasks to avoid circular reference (task->exception->traceback->task) and improve memory performance
                del args, kwargs
                raise _DepNotFoundError() from None
        return (), {}

    def node_to_input_sync(
        accumulated_results: Mapping[base_types.ComputationNode, base_types.Result],
        input_options: Iterable[_ComputationInputSpec],
    ) -> Optional[_ComputationInput]:
        for input_spec in input_options:
            args, kwargs = input_spec
            try:
                return tuple(accumulated_results[arg] for arg in args), {
                    k: accumulated_results[v] for k, v in kwargs.items()
                }
            except KeyError:
                ...
        return None

    async def node_to_input_async(
        accumulated_results: Mapping[
            base_types.ComputationNode, base_types.Result | Awaitable[base_types.Result]
        ],
        input_options: Iterable[_ComputationInputSpec],
    ) -> Optional[_ComputationInput]:
        for input_spec in input_options:
            args_spec, kwargs_spec = input_spec
            if all(
                accumulated_results.get(arg, CG_NO_RESULT) is not CG_NO_RESULT
                for arg in args_spec
            ) and all(
                accumulated_results.get(kwarg, CG_NO_RESULT) is not CG_NO_RESULT
                for kwarg in kwargs_spec.values()
            ):
                try:
                    return await gather(
                        tuple(_to_awaitable(accumulated_results[a]) for a in args_spec),
                        {
                            k: _to_awaitable(accumulated_results[v])
                            for k, v in kwargs_spec.items()
                        },
                    )
                except (
                    _DepNotFoundError,
                    base_types.SkipComputationError,
                    *handled_exceptions,
                ):
                    ...
        return None

    @opt_gamla.after(asyncio.create_task)
    async def await_deps_and_apply(
        accumulated_results: Mapping[
            base_types.ComputationNode, base_types.Result | Awaitable[base_types.Result]
        ],
        node: base_types.ComputationNode,
    ) -> base_types.Result:
        args_kwargs = await node_to_input_async(
            accumulated_results, node_to_computation_input_spec_options(node)
        )
        # We delete the references to the upstream tasks to avoid circular reference (task->exception->traceback->task) and improve memory performance
        del accumulated_results
        if args_kwargs is None:
            raise _DepNotFoundError()

        args, kwargs = args_kwargs
        before = time.perf_counter()
        result = node.func(*args, **kwargs)
        single_node_side_effect(node, result)
        if inspect.isawaitable(result):
            raise Exception(
                f"{node} returned an awaitable result but is not an async function"
            )
        _profile(node, before)
        return result

    @opt_gamla.after(asyncio.create_task)
    async def await_deps_and_await(
        accumulated_results: Mapping[
            base_types.ComputationNode, base_types.Result | Awaitable[base_types.Result]
        ],
        node: base_types.ComputationNode,
    ) -> base_types.Result:
        args_kwargs = await node_to_input_async(
            accumulated_results, node_to_computation_input_spec_options(node)
        )
        # We delete the references to the upstream tasks to avoid circular reference (task->exception->traceback->task) and improve memory performance
        del accumulated_results
        if args_kwargs is None:
            raise _DepNotFoundError()

        args, kwargs = args_kwargs
        before = time.perf_counter()
        result = await node.func(*args, **kwargs)
        single_node_side_effect(node, result)
        _profile(node, before)
        return result

    @opt_gamla.after(asyncio.create_task)
    async def get_deps_and_await(
        accumulated_results: Mapping[base_types.ComputationNode, base_types.Result],
        node: base_types.ComputationNode,
    ) -> base_types.Result:
        args_kwargs = node_to_input_sync(
            accumulated_results, node_to_computation_input_spec_options(node)
        )
        # We delete the references to the upstream tasks to avoid circular reference (task->exception->traceback->task) and improve memory performance
        del accumulated_results
        if args_kwargs is None:
            raise _DepNotFoundError()

        args, kwargs = args_kwargs
        before = time.perf_counter()
        result = await node.func(*args, **kwargs)
        single_node_side_effect(node, result)
        _profile(node, before)
        return result

    def get_deps_and_apply(
        accumulated_results: Mapping[base_types.ComputationNode, base_types.Result],
        node: base_types.ComputationNode,
    ) -> base_types.Result:
        args_kwargs = node_to_input_sync(
            accumulated_results, node_to_computation_input_spec_options(node)
        )
        # We delete the references to the upstream tasks to avoid circular reference (task->exception->traceback->task) and improve memory performance
        del accumulated_results
        if args_kwargs is None:
            raise _DepNotFoundError()

        args, kwargs = args_kwargs
        before = time.perf_counter()
        result = node.func(*args, **kwargs)
        single_node_side_effect(node, result)
        if inspect.isawaitable(result):
            raise Exception(
                f"{node} returned an awaitable result but is not an async function"
            )
        _profile(node, before)
        return result

    all_nodes = graph.get_all_nodes(edges)
    async_nodes = {n for n in all_nodes if asyncio.iscoroutinefunction(n.func)}
    sync = all_nodes - async_nodes
    tf = graph.traverse_forward(edges)
    downstream_from_async = set(gamla.graph_traverse_many(async_nodes, tf))

    async_and_downstream = async_nodes & downstream_from_async
    async_not_downstream = async_nodes - downstream_from_async
    sync_and_downstream = sync & downstream_from_async
    sync_not_downstream = sync - downstream_from_async

    def get_executor(node: base_types.ComputationNode) -> _NodeExecutor:
        if node in async_and_downstream:
            return await_deps_and_await
        if node in async_not_downstream:
            return get_deps_and_await
        if node in sync_and_downstream:
            return await_deps_and_apply
        if node in sync_not_downstream:
            # This is fully sync so it only uses sync results from the mapping, its typing says the whole mapping is sync.
            return get_deps_and_apply  # type: ignore
        raise Exception("no executor found")

    return get_executor


def _run_graph(inputs: dict, handled_exceptions):
    def run_graph(
        nodes: tuple[tuple[base_types.ComputationNode, _NodeExecutor]]
    ) -> _NodeToResults:
        accumulated_results = inputs.copy()
        for node_executor in nodes:
            try:
                accumulated_results[node_executor[0]] = node_executor[1](
                    accumulated_results, node_executor[0]
                )
            except (
                _DepNotFoundError,
                base_types.SkipComputationError,
                *handled_exceptions,
            ):
                pass
        return accumulated_results

    return run_graph


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
) -> Callable[[_NodeToResults, _NodeToResults], _NodeToResults]:
    return gamla.compose(
        gamla.star(to_callable(g, frozenset())), gamla.map(immutables.Map), gamla.pack
    )
