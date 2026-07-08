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




@dataclasses.dataclass(frozen=True)
class ChangeActiveColors:
    """The CG-native CHANGE-COLOR EVENT. A node RETURNS this value to declare the
    set of colors that are active for node-activation skipping. The runner reads it
    out of node results -- it is the ONLY way the active colors are set, so the
    engine never needs to know what colors mean (skills, etc. -- that is entirely
    the caller's concern).

    The colors are opaque hashable tokens, matched against `node_to_colors`. An
    EMPTY set means "this declarer resolved no color" -> alone it activates nothing
    (e.g. a turn that resolved to no skill).

    Declarations COMPOSE: independent subgraphs may each declare the colors they
    resolved (e.g. one declarer per routing context), and the runner activates the
    UNION of every declaration observed in a pass (replacing the caller's initial
    seed). One turn can thus activate several colors at once; a declarer that
    resolved nothing contributes nothing rather than vetoing the others.
    """

    colors: FrozenSet


# The reserved results key under which the colored runner records the turn's SETTLED
# effective active-color set (the union the restart loop converged on). Callers that
# persist state across turns should carry this value forward as the NEXT run's seed
# `active_colors`: it survives the restart intact, unlike any single declarer node,
# which is color-dependent and prunes when the settled colors differ from its own
# declaration (e.g. a mid-flow route change -> the new skill's declarer prunes under
# the newly-active color, so its declaration is lost). Domain-agnostic: just colors.
EFFECTIVE_ACTIVE_COLORS_KEY = "__cg_effective_active_colors__"


class ColorDeclarationsDidNotConverge(Exception):
    """The unioned `ChangeActiveColors` declarations cycled back to an active set
    this run already had -- the declarations depend on the active colors themselves,
    so the restart loop can never settle. Declarers must derive their colors from
    always-on (uncolored) inputs only."""


@dataclasses.dataclass(frozen=True, eq=False)
class NodeActivation:
    """Opt-in spec for runner-level node skipping (per-node "coloring").

    The engine stays domain-agnostic: it knows only COLORS (opaque tokens) and the
    `ChangeActiveColors` event. The caller colors the nodes and emits the event from
    its own nodes; nothing skill/route/VIC-specific reaches the engine.

    SKIP RULE -- when a node's color-set is DISJOINT from the active colors:
      * if the node is in `boundary_defaults` (a colored->uncolored frontier that
        feeds shared recombination, e.g. an aggregation/terminal), its typed-empty
        default value is returned so downstream make_and/aggregation stays
        well-formed (the per-node, build-time, correctly-typed sentinel);
      * otherwise `_DepNotFoundError` is raised so the node prunes and the graph
        reducer's `{**prev, **computed}` merge latches its previous value for free.

    ACTIVE COLORS: the CALLER provides an initial color set when it starts the run
    (the `active_colors` argument of the compiled reducer; optional). The run then
    proceeds layer by layer skipping colored nodes whose color is not in the active
    set -- when no initial color is given, ONLY the no-color (uncolored) nodes run.
    If `ChangeActiveColors` events then appear in node results whose UNION differs
    from the active set, the run STARTS OVER with that union as the new active set
    (nodes that were skipped may now need to run); only the colored nodes +
    everything downstream of them are recomputed, the rest (e.g. an upstream routing
    model call) is carried over so it is never repeated. The run restarts until the
    unioned declaration stabilizes (normally once); a union cycling back to an
    earlier active set raises `ColorDeclarationsDidNotConverge`.
    """

    node_to_colors: Mapping[base_types.ComputationNode, FrozenSet]
    boundary_defaults: Mapping[base_types.ComputationNode, Any] = dataclasses.field(
        default_factory=dict
    )


def _to_callable_with_side_effect_for_single_and_multiple(
    single_node_side_effect: _SingleNodeSideEffect,
    all_nodes_side_effect: Callable,
    input_graph: base_types.GraphType,
    handled_exceptions: Tuple[Type[Exception], ...],
    node_activation: Optional[NodeActivation] = None,
) -> Callable[[_NodeToResults, _NodeToResults], _NodeToResults]:
    edges = _merge_edges_pointing_to_terminals(input_graph).edges
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

    # Per-node skipping ("coloring"), opt-in by construction: it applies only when a
    # NodeActivation is supplied (via `to_callable_with_coloring` / the side-effect
    # curry with a trailing activation). The runner
    # itself does the skip (no executor wrapping): a colored node whose color is not
    # active returns its boundary default (frontier) or is pruned (interior). Plain
    # `to_callable` passes empty maps -> the runner runs the whole graph (zero
    # behavior change for every existing caller). `color_dependent` = colored nodes
    # + everything downstream of them (recomputed on a restart; the rest carries
    # over so upstream async work is never repeated).
    node_to_colors: Mapping = (
        node_activation.node_to_colors if node_activation is not None else {}
    )
    boundary_defaults: Mapping = (
        node_activation.boundary_defaults if node_activation is not None else {}
    )
    color_dependent: FrozenSet = (
        frozenset(
            gamla.graph_traverse_many(
                tuple(node_to_colors), graph.traverse_forward(edges)
            )
        )
        if node_to_colors
        else frozenset()
    )

    translate_source_to_placeholder = opt_gamla.compose_left(
        opt_gamla.keyfilter(gamla.contains(future_sources)),
        opt_gamla.keymap(future_source_to_placeholder.__getitem__),
    )
    all_node_side_effects_on_edges = gamla.side_effect(all_nodes_side_effect(edges))

    if is_async:

        async def final_runner(sources_to_values, active_colors=None):
            inputs = translate_source_to_placeholder(sources_to_values)
            all_results = await _run_graph_async(
                inputs,
                handled_exceptions,
                topological_sorted_nodes,
                node_to_colors,
                boundary_defaults,
                color_dependent,
                active_colors,
            )

            return all_node_side_effects_on_edges(all_results)

    else:

        def final_runner(sources_to_values, active_colors=None):
            return all_node_side_effects_on_edges(
                _run_graph(
                    translate_source_to_placeholder(sources_to_values),
                    handled_exceptions,
                    topological_sorted_nodes,
                    node_to_colors,
                    boundary_defaults,
                    color_dependent,
                    active_colors,
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


def _schedule_node(node_executor, node_to_task_or_result, handled_exceptions):
    try:
        node_to_task_or_result[node_executor[0]] = node_executor[1](
            node_to_task_or_result, node_executor[0]
        )
    except (_DepNotFoundError, base_types.SkipComputationError, *handled_exceptions):
        pass


def _next_active_colors(
    topological_sorted_nodes, results, active: FrozenSet, seen_active_sets: Set
) -> Optional[FrozenSet]:
    """Decide the restart loop's next active set from the pass's results.

    Declarations COMPOSE: independent declarers each contribute the colors they
    resolved, and the pass's active set is the UNION of every `ChangeActiveColors`
    observed (an empty declaration contributes nothing). Returns the NEW active set
    (the caller restarts with it), or None when the run has settled -- nothing was
    declared (the seed stands) or the union already matches `active`. A union that
    revisits an earlier active set means the declarations depend on the active
    colors themselves, so the loop can never settle -> raises
    `ColorDeclarationsDidNotConverge`. Mutates `seen_active_sets`."""
    declarations = frozenset(
        value
        for node_executor in topological_sorted_nodes
        for value in (results.get(node_executor[0], CG_NO_RESULT),)
        if isinstance(value, ChangeActiveColors)
    )
    if not declarations:
        return None
    observed = frozenset().union(*(d.colors for d in declarations))
    if observed == active:
        return None
    if observed in seen_active_sets:
        raise ColorDeclarationsDidNotConverge(
            f"{sorted(map(str, observed))} was already active this run"
        )
    seen_active_sets.add(observed)
    return observed


def _drop_color_dependent(results, color_dependent: FrozenSet) -> None:
    # Start a restarted pass clean: forget the color-dependent results (they are
    # recomputed under the new active set); everything else (e.g. upstream routing /
    # async work) carries over so it is never repeated.
    for node in tuple(results):
        if node in color_dependent:
            del results[node]


async def _run_graph_async(
    inputs,
    handled_exceptions,
    topological_sorted_nodes,
    node_to_colors,
    boundary_defaults,
    color_dependent,
    initial_colors,
):
    node_to_task_or_result = inputs.copy()
    unhandled_exception = None
    skip_exceptions = (
        _DepNotFoundError,
        base_types.SkipComputationError,
        *handled_exceptions,
    )
    # The caller provides the initial active colors; no initial color -> the empty
    # set, disjoint from every color, so only the no-color (uncolored) nodes run.
    active = initial_colors if initial_colors is not None else frozenset()

    def _schedule_or_skip(node_executor):
        node = node_executor[0]
        colors = node_to_colors.get(node)
        if colors and colors.isdisjoint(active):
            # colored but not active: frontier -> boundary default; interior -> prune.
            if node in boundary_defaults:
                node_to_task_or_result[node] = boundary_defaults[node]
            return
        _schedule_node(node_executor, node_to_task_or_result, handled_exceptions)

    async def _gather_pending():
        # Await every pending future in `node_to_task_or_result` in ONE gather, folding
        # each result back in place: success -> value, skip-exception -> pruned (absent).
        # Returns the first unhandled (non-skip) exception, or None. Every future is
        # awaited regardless of outcome, so no task's exception goes unretrieved. Shared
        # by the restart loop (which reads results before checking for a color change)
        # and the final harvest below.
        pending = [
            node
            for node, value in node_to_task_or_result.items()
            if asyncio.isfuture(value)
        ]
        first_unhandled = None
        for node, result in zip(
            pending,
            await asyncio.gather(
                *(node_to_task_or_result[n] for n in pending), return_exceptions=True
            ),
        ):
            if not isinstance(result, Exception):
                node_to_task_or_result[node] = result
            elif isinstance(result, skip_exceptions):
                node_to_task_or_result.pop(node, None)
            elif first_unhandled is None:
                first_unhandled = result
        return first_unhandled

    try:
        # Same restart loop as the sync runner, but a node's result is a Task we must
        # await. We schedule the whole pass first (the tasks run CONCURRENTLY), then
        # gather them together (not one-by-one) before checking for a color change.
        seen_active_sets = {active}
        while True:
            for node_executor in topological_sorted_nodes:
                if node_executor[0] in node_to_task_or_result:
                    continue  # carried over (color-independent) or already scheduled
                _schedule_or_skip(node_executor)
            if not color_dependent:
                break  # no colored nodes -> nothing to restart for
            harvested = await _gather_pending()
            if harvested is not None:
                raise harvested
            observed = _next_active_colors(
                topological_sorted_nodes,
                node_to_task_or_result,
                active,
                seen_active_sets,
            )
            if observed is None:
                break  # settled: nothing declared, or the union matches
            # A new color set was declared -> START OVER with it.
            active = observed
            _drop_color_dependent(node_to_task_or_result, color_dependent)
    except Exception as exc:
        unhandled_exception = exc
    finally:
        # Harvest any futures still pending (the whole graph when uncolored / the
        # color-independent ones when the restart loop broke without gathering).
        harvested = await _gather_pending()
        if unhandled_exception is None:
            unhandled_exception = harvested
        if unhandled_exception:
            # this trick avoids cyclic reference and garbage collection issues
            try:
                raise unhandled_exception from unhandled_exception
            except Exception as e:
                del node_to_task_or_result
                del unhandled_exception
                raise e from e
    if color_dependent:
        # Record the SETTLED active set so the caller can seed it next turn (see
        # EFFECTIVE_ACTIVE_COLORS_KEY): survives restarts that the declarer nodes do not.
        node_to_task_or_result[EFFECTIVE_ACTIVE_COLORS_KEY] = ChangeActiveColors(active)
    return node_to_task_or_result


def _run_graph(
    inputs: dict,
    handled_exceptions,
    topological_sorted_nodes: tuple[tuple[base_types.ComputationNode, _NodeExecutor]],
    node_to_colors,
    boundary_defaults,
    color_dependent,
    initial_colors,
) -> _NodeToResults:
    accumulated_results = inputs.copy()
    # The caller provides the initial active colors (no color -> empty set -> only
    # no-color nodes run); a ChangeActiveColors event that declares a different set
    # starts the run over, recomputing only the color-dependent nodes and carrying
    # the rest over. With no coloring (`to_callable`) the loop runs the whole graph
    # once, exactly as before.
    active = initial_colors if initial_colors is not None else frozenset()
    # Same restart loop as the async runner: run the WHOLE pass, then union the
    # declarations (a later declarer in the toposort must be seen before deciding
    # the active set), and start over if a new color set was declared.
    seen_active_sets = {active}
    while True:
        for node_executor in topological_sorted_nodes:
            node = node_executor[0]
            if node in accumulated_results:
                continue  # carried over (color-independent) or already run this pass
            colors = node_to_colors.get(node)
            if colors and colors.isdisjoint(active):
                # colored but not active: frontier -> boundary default; interior -> prune.
                if node in boundary_defaults:
                    accumulated_results[node] = boundary_defaults[node]
                continue
            _schedule_node(node_executor, accumulated_results, handled_exceptions)
        if not color_dependent:
            break  # no colored nodes -> nothing to restart for
        observed = _next_active_colors(
            topological_sorted_nodes, accumulated_results, active, seen_active_sets
        )
        if observed is None:
            break  # settled: nothing declared, or the union matches
        # A new color set was declared -> start over with it.
        active = observed
        _drop_color_dependent(accumulated_results, color_dependent)
    if color_dependent:
        # Record the SETTLED active set so the caller can seed it next turn (see
        # EFFECTIVE_ACTIVE_COLORS_KEY): survives restarts that the declarer nodes do not.
        accumulated_results[EFFECTIVE_ACTIVE_COLORS_KEY] = ChangeActiveColors(active)
    return accumulated_results


def _graph_reducer(graph_callable):
    # `active_colors` (optional) is the initial color set the caller provides when
    # it starts the run; ignored unless the graph has node-activation coloring.
    def reducer(
        prev: _NodeToResults, sources: _NodeToResults, active_colors=None
    ) -> _NodeToResults:
        return {**prev, **(graph_callable({**prev, **sources}, active_colors))}

    return reducer


def _async_graph_reducer(graph_callable):
    async def reducer(
        prev: _NodeToResults, sources: _NodeToResults, active_colors=None
    ) -> _NodeToResults:
        return {**prev, **(await graph_callable({**prev, **sources}, active_colors))}

    return reducer


to_callable_with_side_effect = gamla.curry(
    _to_callable_with_side_effect_for_single_and_multiple
)(_type_check)

# Use the second line if you want to see the winning path in the computation graph (a little slower).
to_callable = to_callable_with_side_effect(gamla.just(gamla.just(None)))
# to_callable = to_callable_with_side_effect(graphviz.computation_trace('utterance_computation.dot'))


def to_callable_with_coloring(
    edges: base_types.GraphType,
    handled_exceptions: Tuple[Type[Exception], ...],
) -> Callable[[_NodeToResults, _NodeToResults], _NodeToResults]:
    """Like `to_callable`, but with per-node skipping ("coloring") derived ENTIRELY
    from the author tags carried on the graph's node funcs (colors and typed-empties;
    observer marks are diagnostic-only and do not affect the activation). The caller
    only tags the graph at composition time (`coloring.add_colors` /
    `add_colors(..., empty=...)` / `coloring.pin_core`) and
    emits the `run.ChangeActiveColors` event from its own node; this derives the
    `NodeActivation` generically and compiles with it. The compiled reducer takes an
    extra optional `active_colors` argument (the initial color set the caller
    provides at run start). Uncolored graphs / plain `to_callable` behave identically
    to before.

    Callers that already hold a `NodeActivation` (the engine's own tests, dev
    diagnostics) compile it directly via `to_callable_with_side_effect`."""
    # Local import: `coloring` imports `run`, so importing it at module load would
    # be circular.
    from computation_graph.composers import coloring

    return _to_callable_with_side_effect_for_single_and_multiple(
        _type_check,
        gamla.just(gamla.just(None)),
        edges,
        handled_exceptions,
        coloring.build_node_activation_from_edges(edges),
    )


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
