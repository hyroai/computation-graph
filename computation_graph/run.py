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
from computation_graph.base_types import GraphType

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
            base_types.ComputationNode, base_types.Result | Awaitable[base_types.Result]
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
    toposort.toposort,
    # Make async functions come first in each layer so they'll start running before all the sync functions
    opt_gamla.maptuple(
        gamla.sort_by(lambda n: 0 if inspect.iscoroutinefunction(n.func) else 1)
    ),
    gamla.concat,
    tuple,
)


def _type_check(node: base_types.ComputationNode, result):
    return_typing = typing.get_type_hints(node.func).get("return", None)
    if return_typing:
        try:
            typeguard.check_type(result, return_typing)
        except (typeguard.TypeCheckError, TypeError) as e:
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


_group_by_is_future = opt_gamla.groupby(lambda k_v: asyncio.isfuture(k_v[1]))


_is_graph_async = opt_gamla.compose_left(
    opt_gamla.mapcat(lambda edge: (edge.source, *edge.args)),
    opt_gamla.remove(gamla.equals(None)),
    opt_gamla.map(base_types.node_implementation),
    gamla.anymap(inspect.iscoroutinefunction),
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
    return gamla.pipe(
        g,
        gamla.attrgetter("edges"),
        gamla.groupby(gamla.attrgetter("destination")),
        gamla.itemmap(
            gamla.compose_left(
                gamla.star(
                    lambda dest, edges_for_dest: (
                        dest,
                        (
                            graph.merge_graphs(
                                composers.make_or(
                                    opt_gamla.maptuple(base_types.edge_source)(
                                        edges_for_dest
                                    ),
                                    merge_fn=(aggregate := lambda args: args),
                                ),
                                composers.compose_left_unary(aggregate, dest),
                                sink_node_or_graph=graph.make_computation_node(
                                    aggregate
                                ),
                            ).edges
                            if dest.is_terminal
                            else edges_for_dest
                        ),
                    )
                )
            )
        ),
        dict.values,
        gamla.concat,
        frozenset,
        lambda edges: GraphType(edges, g.sink),
    )


def _to_callable_with_side_effect_for_single_and_multiple(
    single_node_side_effect: _SingleNodeSideEffect,
    all_nodes_side_effect: Callable,
    graph: base_types.GraphType,
    handled_exceptions: Tuple[Type[Exception], ...],
) -> Callable[[_NodeToResults, _NodeToResults], _NodeToResults]:
    edges = _merge_edges_pointing_to_terminals(graph).edges
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
        gamla.side_effect(base_types.assert_no_unwanted_ambiguity_on_edges),
    )
    is_async = _is_graph_async(edges)
    placeholder_to_future_source = opt_gamla.pipe(
        future_source_to_placeholder, gamla.itemmap(lambda k_v: (k_v[1], k_v[0]))
    )
    get_node_executor = _make_get_node_executor(
        edges, handled_exceptions, single_node_side_effect
    )

    topological_sorted_nodes = opt_gamla.pipe(
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

    if is_async and os.getenv("CG_ASYNC_INLINE") == "1":
        if not _EAGER_TASKS_NATIVE:
            raise RuntimeError(
                "CG_ASYNC_INLINE requires Python 3.12+ (native eager asyncio tasks)."
            )
        inline_plan = _build_inline_plan(
            edges, tuple(n for n, _ in topological_sorted_nodes)
        )

        async def final_runner(sources_to_values):
            inputs = translate_source_to_placeholder(sources_to_values)
            all_results = await _run_graph_async_inline(
                inputs, handled_exceptions, inline_plan, single_node_side_effect
            )
            return all_node_side_effects_on_edges(all_results)

    elif is_async:

        async def final_runner(sources_to_values):
            inputs = translate_source_to_placeholder(sources_to_values)
            all_results = await _run_graph_async(
                inputs, handled_exceptions, topological_sorted_nodes
            )

            return all_node_side_effects_on_edges(all_results)

    else:

        def final_runner(sources_to_values):
            return all_node_side_effects_on_edges(
                _run_graph(
                    translate_source_to_placeholder(sources_to_values),
                    handled_exceptions,
                    topological_sorted_nodes,
                )
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
                # Resolve inputs in place: await only the still-pending ones
                # (already-scheduled tasks, so concurrency is preserved) and pass
                # concrete values straight through. This avoids allocating a
                # resolved Future per concrete input and the asyncio.gather
                # machinery; the comprehension does zero awaits when nothing is
                # pending, so it completes without suspending.
                args = [accumulated_results[a] for a in args_spec]
                kwargs = {k: accumulated_results[v] for k, v in kwargs_spec.items()}
                try:
                    return (
                        tuple([await x if inspect.isawaitable(x) else x for x in args]),
                        {
                            k: (await x if inspect.isawaitable(x) else x)
                            for k, x in kwargs.items()
                        },
                    )
                except (
                    _DepNotFoundError,
                    base_types.SkipComputationError,
                    *handled_exceptions,
                ):
                    # Drop refs to the upstream tasks to avoid a
                    # task -> exception -> traceback -> task reference cycle.
                    del args, kwargs
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
    async_nodes = {n for n in all_nodes if inspect.iscoroutinefunction(n.func)}
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


async def _run_graph_async(inputs, handled_exceptions, topological_sorted_nodes):
    node_to_task_or_result = inputs.copy()
    unhandled_exception = None
    try:
        for node_executor in topological_sorted_nodes:
            try:
                node_to_task_or_result[node_executor[0]] = node_executor[1](
                    node_to_task_or_result, node_executor[0]
                )
            except (
                _DepNotFoundError,
                base_types.SkipComputationError,
                *handled_exceptions,
            ):
                pass
    except Exception as exc:
        unhandled_exception = exc
    finally:
        results_by_is_async = _group_by_is_future(node_to_task_or_result.items())
        async_results = tuple(zip(*results_by_is_async.get(True, ())))
        sync_results = dict(results_by_is_async.get(False, ()))

        all_results = sync_results
        if async_results:
            for node, node_result in zip(
                async_results[0],
                await asyncio.gather(*async_results[1], return_exceptions=True),
            ):
                task_e = node_to_task_or_result[node].exception()
                if not task_e:
                    all_results[node] = node_result
                elif not unhandled_exception and not isinstance(
                    task_e,
                    (
                        _DepNotFoundError,
                        base_types.SkipComputationError,
                        *handled_exceptions,
                    ),
                ):
                    unhandled_exception = task_e
        if unhandled_exception:
            # this trick avoids cyclic reference and garbage collection issues
            try:
                raise unhandled_exception from unhandled_exception
            except Exception as e:
                del node_to_task_or_result
                del unhandled_exception
                raise e from e
    return all_results


def _run_graph(
    inputs: dict,
    handled_exceptions,
    topological_sorted_nodes: tuple[tuple[base_types.ComputationNode, _NodeExecutor]],
) -> _NodeToResults:
    accumulated_results = inputs.copy()
    for node_executor in topological_sorted_nodes:
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


# The inline scheduler relies on eager tasks (run a coroutine synchronously to
# its first real suspension) so a short-circuiting async node completes
# immediately and its downstream collapses to inline execution. Eager tasks are
# native on Python 3.12+ only; the runner is guarded on this at build time.
_EAGER_TASKS_NATIVE = hasattr(asyncio, "eager_task_factory")


# Types for the inline runner. During a run each node maps to either a plain
# result or a future (a still-pending task, or a finished task that may carry a
# handled/unhandled exception).
_HandledExceptions = Tuple[Type[BaseException], ...]
_InlineResults = Dict[base_types.ComputationNode, Any]
_InputOptions = Tuple[_ComputationInputSpec, ...]
_DepStatus = Tuple[str, Any]  # (kind, payload): value/pending/pruned/error
_InlinePlan = Tuple[Tuple[base_types.ComputationNode, bool, _InputOptions], ...]


def _eager_task(
    coro: Awaitable[base_types.Result], loop: asyncio.AbstractEventLoop
) -> asyncio.Future:
    """Start `coro` eagerly: run it synchronously up to its first real
    suspension. Returns a future that is ALREADY done if the coroutine completed
    (or raised) synchronously (the short-circuit case that makes downstream
    nodes inline-eligible), else a Task that resolves when it finishes. Requires
    Python 3.12+ (guarded where the inline runner is selected)."""
    return asyncio.Task(coro, loop=loop, eager_start=True)


# ---- inline-runner helpers (module-level so the driver loop stays readable) ---
def _inline_dep_status(value_or_future: Any, handled: _HandledExceptions) -> _DepStatus:
    """Classify a stored dependency as ('value', x) | ('pending', None) |
    ('pruned', None) | ('error', exc). A handled-exception or cancelled task
    counts as pruned (no value); an unhandled exception is an 'error' that must
    propagate."""
    if not asyncio.isfuture(value_or_future):
        return ("value", value_or_future)
    if not value_or_future.done():
        return ("pending", None)
    if value_or_future.cancelled():
        return ("pruned", None)
    exception = value_or_future.exception()
    if exception is None:
        return ("value", value_or_future.result())
    return ("pruned", None) if isinstance(exception, handled) else ("error", exception)


def _inline_unwrap(
    results: _InlineResults, node: base_types.ComputationNode
) -> base_types.Result:
    """Read a resolved dependency's value (a finished future, or a plain value)."""
    value = results[node]
    return value.result() if asyncio.isfuture(value) else value


def _inline_present(
    results: _InlineResults,
    node: base_types.ComputationNode,
    handled: _HandledExceptions,
) -> bool:
    """True if `node` produced a value or is still pending -- i.e. an option
    using it is still viable. Absent (pruned) or handled-failed nodes are not."""
    return node in results and _inline_dep_status(results[node], handled)[0] in (
        "value",
        "pending",
    )


async def _inline_resolve_async(
    results: _InlineResults,
    input_options: _InputOptions,
    handled: _HandledExceptions,
) -> Optional[_ComputationInput]:
    """Faithful copy of `node_to_input_async`: try input options in priority
    order, await the first whose deps are all present, and fall through to the
    next on a handled raise. Returns (args, kwargs) or None."""
    for args_spec, kwargs_spec in input_options:
        if all(_inline_present(results, dep, handled) for dep in args_spec) and all(
            _inline_present(results, dep, handled) for dep in kwargs_spec.values()
        ):
            try:
                gathered = await asyncio.gather(
                    *(_to_awaitable(results[dep]) for dep in args_spec),
                    *(_to_awaitable(results[dep]) for dep in kwargs_spec.values()),
                )
            except handled:
                continue
            n_args = len(args_spec)
            return tuple(gathered[:n_args]), dict(
                zip(kwargs_spec.keys(), gathered[n_args:])
            )
    return None


def _inline_fast_resolve(
    results: _InlineResults,
    input_options: _InputOptions,
    handled: _HandledExceptions,
) -> tuple:
    """Decide a node synchronously, without awaiting:
    ('run', args, kwargs) | ('defer',) | ('prune',) | ('error', exc).
    Input options are tried in priority order; the outcome is set by the FIRST
    non-value dependency in the first not-yet-failed option (a pending dep ->
    defer, a pruned dep -> try the next option, an unhandled error -> propagate).
    Only when an option's deps are all resolved values do we run it now."""
    for args_spec, kwargs_spec in input_options:
        blocker: Optional[str] = None  # set on the first non-value dependency
        for dep in (*args_spec, *kwargs_spec.values()):
            kind, payload = (
                ("pruned", None)
                if dep not in results
                else _inline_dep_status(results[dep], handled)
            )
            if kind == "value":
                continue
            if kind == "error":
                return ("error", payload)
            blocker = kind
            break
        if blocker == "pending":
            return ("defer",)
        if blocker == "pruned":
            continue  # this option is dead -> try the next (lower priority) one
        # all deps are resolved values
        return (
            "run",
            tuple(_inline_unwrap(results, dep) for dep in args_spec),
            {key: _inline_unwrap(results, dep) for key, dep in kwargs_spec.items()},
        )
    return ("prune",)


async def _inline_compute_deferred(
    results: _InlineResults,
    node: base_types.ComputationNode,
    is_async_node: bool,
    input_options: _InputOptions,
    handled: _HandledExceptions,
    side_effect: _SingleNodeSideEffect,
) -> base_types.Result:
    """The task path for a node that depends on a still-pending async result:
    await its inputs (with priority fallback) then run it."""
    resolved_inputs = await _inline_resolve_async(results, input_options, handled)
    if resolved_inputs is None:
        raise _DepNotFoundError()
    args, kwargs = resolved_inputs
    before = time.perf_counter()
    result = node.func(*args, **kwargs)
    if is_async_node:
        result = await result
    side_effect(node, result)
    _profile(node, before)
    return result


async def _inline_run_async(
    node: base_types.ComputationNode,
    args: Tuple[base_types.Result, ...],
    kwargs: Dict[str, base_types.Result],
    side_effect: _SingleNodeSideEffect,
) -> base_types.Result:
    """Run an async node whose inputs are already resolved (eager-started so it
    overlaps other async work and short-circuits inline when it doesn't await)."""
    before = time.perf_counter()
    result = await node.func(*args, **kwargs)
    side_effect(node, result)
    _profile(node, before)
    return result


def _build_inline_plan(
    edges: base_types.GraphType,
    ordered_nodes: Tuple[base_types.ComputationNode, ...],
) -> _InlinePlan:
    """Per-node (node, is_async, input_options) for the inline async runner, in
    topological order. Same input-spec options as `_make_get_node_executor`,
    exposed as data."""
    node_to_incoming_edges = functools.cache(graph.get_incoming_edges_for_node(edges))

    def input_options(node: base_types.ComputationNode) -> _InputOptions:
        edges_by_key: Dict[str, list] = {}
        for edge in node_to_incoming_edges(node):
            edges_by_key.setdefault(base_types.edge_key(edge), []).append(edge)
        for key in edges_by_key:
            edges_by_key[key].sort(key=base_types.edge_priority)
        return tuple(
            _node_incoming_edges_to_input_spec(edge_combination)
            for edge_combination in itertools.product(*edges_by_key.values())
        )

    return tuple(
        (node, asyncio.iscoroutinefunction(node.func), input_options(node))
        for node in ordered_nodes
    )


async def _run_graph_async_inline(
    inputs: _InlineResults,
    handled_exceptions: _HandledExceptions,
    plan: _InlinePlan,
    single_node_side_effect: _SingleNodeSideEffect,
) -> _NodeToResults:
    """Async runner that executes a node INLINE when its dependencies are
    already resolved values, instead of task-wrapping every node downstream of
    an async node. Behaviour-equivalent to `_run_graph_async`; opt-in via the
    CG_ASYNC_INLINE env flag. Only nodes that truly depend on a still-pending
    async result take the task path; everything else runs inline.

    Semantics preserved: priority option-fallback on handled raises (the
    deferred path mirrors `node_to_input_async`), side-effect/_profile hooks,
    the sync isawaitable guard, concurrency of independent async ops (eager
    tasks), and the unhandled-exception cyclic-ref cleanup dance."""
    handled = (_DepNotFoundError, base_types.SkipComputationError, *handled_exceptions)
    loop = asyncio.get_running_loop()
    results: _InlineResults = inputs.copy()
    tasks: list = []
    unhandled: Optional[BaseException] = None

    try:
        for node, is_async_node, input_options in plan:
            kind, *rest = _inline_fast_resolve(results, input_options, handled)
            if kind == "prune":
                continue
            if kind == "error":
                unhandled = rest[0]
                break
            if kind == "defer":
                # depends on a still-pending async result -> task path
                task = _eager_task(
                    _inline_compute_deferred(
                        results, node, is_async_node, input_options, handled,
                        single_node_side_effect,
                    ),
                    loop,
                )
                results[node] = task
                tasks.append(task)
                continue
            # kind == "run": all deps are resolved values
            args, kwargs = rest
            if is_async_node:
                task = _eager_task(
                    _inline_run_async(node, args, kwargs, single_node_side_effect),
                    loop,
                )
                results[node] = task
                tasks.append(task)
            else:
                before = time.perf_counter()
                try:
                    result = node.func(*args, **kwargs)  # INLINE -- no task
                except handled:
                    continue  # produced no value
                except Exception as exc:  # noqa: BLE001
                    unhandled = exc
                    break
                single_node_side_effect(node, result)
                if inspect.isawaitable(result):
                    raise Exception(
                        f"{node} returned an awaitable result but is not an async function"
                    )
                _profile(node, before)
                results[node] = result
    except Exception as exc:  # noqa: BLE001
        unhandled = exc
    finally:
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        all_results: _NodeToResults = {}
        for node, value in results.items():
            if asyncio.isfuture(value):
                if value.cancelled():
                    continue
                task_exc = value.exception()
                if task_exc is None:
                    all_results[node] = value.result()
                elif isinstance(task_exc, handled):
                    continue
                elif unhandled is None:
                    unhandled = task_exc
            else:
                all_results[node] = value
        if unhandled is not None:
            # this trick avoids cyclic reference and garbage collection issues
            try:
                raise unhandled from unhandled
            except Exception as exc:
                del results
                del tasks
                del unhandled
                raise exc from exc
    return all_results


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
    node_to_incoming_edges: Callable[
        [base_types.ComputationNode], FrozenSet[base_types.ComputationEdge]
    ]
) -> Callable[[base_types.ComputationNode], bool]:
    return gamla.compose_left(
        graph.unbound_signature(node_to_incoming_edges),
        signature.parameters,
        gamla.len_equals(0),
    )


def _assert_composition_is_valid(g: typing.Iterable[base_types.ComputationEdge]):
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
