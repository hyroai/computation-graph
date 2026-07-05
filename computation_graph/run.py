import asyncio
import collections
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


def _sync_fusion_enabled() -> bool:
    return os.getenv(base_types.COMPUTATION_GRAPH_SYNC_FUSION_ENV_KEY, "0") not in (
        "0",
        "false",
        "False",
        "",
    )


class _FusedSyncChain:
    """A maximal linear run of synchronous nodes downstream of async nodes.

    The whole chain executes inside a single asyncio Task instead of one Task
    per node. `published` are the chain nodes whose result is consumed outside
    the chain; each gets a Future in the results mapping so external consumers
    can await it (resolved the moment the node computes or skips).
    """

    __slots__ = ("nodes", "published")

    def __init__(
        self,
        nodes: Tuple[base_types.ComputationNode, ...],
        published: Tuple[base_types.ComputationNode, ...],
    ):
        self.nodes = nodes
        self.published = published

    def __repr__(self):
        return f"sync_chain[{'->'.join(map(str, self.nodes))}]"


def _sync_chains(
    edges: base_types.GraphType,
    sync_and_downstream: Set[base_types.ComputationNode],
) -> Tuple[Tuple[base_types.ComputationNode, ...], ...]:
    """Maximal linear chains in the subgraph induced on `sync_and_downstream`.

    A link u->v requires v to be u's only consumer within the induced subgraph
    and u to be v's only producer within it. Chain nodes may still have inputs
    from and outputs to nodes outside the chain (async nodes, sources, other
    chains); those stay external. Only chains of length >= 2 are returned.
    """
    successors = collections.defaultdict(set)
    predecessors = collections.defaultdict(set)
    for edge in edges:
        destination = edge.destination
        if destination not in sync_and_downstream:
            continue
        for source in base_types.edge_sources(edge):
            if source in sync_and_downstream:
                successors[source].add(destination)
                predecessors[destination].add(source)
    next_in_chain = {
        node: gamla.head(consumers)
        for node, consumers in successors.items()
        if len(consumers) == 1 and len(predecessors[gamla.head(consumers)]) == 1
    }
    linked = set(next_in_chain.values())
    chains = []
    for head in next_in_chain:
        if head in linked:
            continue
        chain = [head]
        while chain[-1] in next_in_chain:
            chain.append(next_in_chain[chain[-1]])
        chains.append(tuple(chain))
    return tuple(chains)


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


_group_by_is_future = opt_gamla.groupby(lambda k_v: asyncio.isfuture(k_v[1]))


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
    is_debug = os.getenv(base_types.COMPUTATION_GRAPH_DEBUG_ENV_KEY) is not None
    single_node_side_effect = (
        single_node_side_effect if is_debug else (lambda node, result: result)
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
    topological_sorted_nodes = opt_gamla.pipe(
        edges,
        _toposort_nodes,
        gamla.remove(gamla.contains(placeholder_to_future_source)),
        tuple,
    )
    fuse_sync_chains = is_async and _sync_fusion_enabled()
    execution_units = _make_execution_units(
        edges,
        handled_exceptions,
        single_node_side_effect,
        topological_sorted_nodes,
        fuse_sync_chains,
    )

    translate_source_to_placeholder = opt_gamla.compose_left(
        opt_gamla.keyfilter(gamla.contains(future_sources)),
        opt_gamla.keymap(future_source_to_placeholder.__getitem__),
    )
    all_node_side_effects_on_edges = gamla.side_effect(all_nodes_side_effect(edges))

    if is_async:

        async def final_runner(sources_to_values):
            inputs = translate_source_to_placeholder(sources_to_values)
            all_results = await _run_graph_async(
                inputs, handled_exceptions, execution_units
            )

            return all_node_side_effects_on_edges(all_results)

        if fuse_sync_chains and is_debug:
            # Debug equivalence check: also run the graph without fusion and
            # compare. Note this runs every node twice per evaluation.
            unfused_units = _make_execution_units(
                edges,
                handled_exceptions,
                single_node_side_effect,
                topological_sorted_nodes,
                False,
            )
            fused_runner = final_runner

            async def final_runner(sources_to_values):  # noqa: F811
                fused_results = await fused_runner(sources_to_values)
                unfused_results = all_node_side_effects_on_edges(
                    await _run_graph_async(
                        translate_source_to_placeholder(sources_to_values),
                        handled_exceptions,
                        unfused_units,
                    )
                )
                _assert_fusion_equivalence(fused_results, unfused_results)
                return fused_results

    else:

        def final_runner(sources_to_values):
            return all_node_side_effects_on_edges(
                _run_graph(
                    translate_source_to_placeholder(sources_to_values),
                    handled_exceptions,
                    execution_units,
                )
            )

    return (_async_graph_reducer if is_async else _graph_reducer)(final_runner)


def _assert_fusion_equivalence(fused: _NodeToResults, unfused: _NodeToResults):
    if set(fused) != set(unfused):
        raise AssertionError(
            "sync chain fusion changed the set of computed nodes."
            f" only fused: {set(fused) - set(unfused)},"
            f" only unfused: {set(unfused) - set(fused)}"
        )
    for node in fused:
        try:
            equal = bool(fused[node] == unfused[node])
        except Exception:
            continue
        if not equal:
            # Not an assertion: nondeterministic nodes and identity-based
            # equality produce false positives here.
            logging.warning(
                f"sync chain fusion: differing results for {node} (may be nondeterminism or identity equality)"
            )


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


def _make_execution_units(
    edges,
    handled_exceptions,
    single_node_side_effect: _SingleNodeSideEffect,
    topological_sorted_nodes: Tuple[base_types.ComputationNode, ...],
    fuse_sync_chains: bool,
) -> Tuple[Tuple[Any, Callable], ...]:
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

    if not fuse_sync_chains:
        return opt_gamla.maptuple(opt_gamla.pair_right(get_executor))(
            topological_sorted_nodes
        )

    handled = (base_types.SkipComputationError, *handled_exceptions)
    # Nodes whose value in the results mapping is never a Future: inputs
    # (placeholders and sources are removed from the executed topological
    # order) and nodes computed by direct call in the walk. Chain members are
    # added per chain: by execution order they are plain values (or absent)
    # by the time a later chain node reads them.
    statically_concrete = sync_not_downstream | (
        all_nodes - set(topological_sorted_nodes)
    )

    def make_chain_executor(chain: _FusedSyncChain):
        chain_set = frozenset(chain.nodes)
        concrete_for_chain = statically_concrete | chain_set

        def split_input_options(node):
            """Prefix of priority-ordered options resolvable with plain dict
            lookups; from the first option that may reference a Future on, the
            rest is handled by node_to_input_async (which resolves concrete
            values too, so order is preserved)."""
            options = node_to_computation_input_spec_options(node)
            concrete_options: list = []
            for index, option in enumerate(options):
                args_spec, kwargs_spec = option
                if all(
                    source in concrete_for_chain
                    for source in (*args_spec, *kwargs_spec.values())
                ):
                    concrete_options.append(option)
                else:
                    return tuple(concrete_options), options[index:]
            return tuple(concrete_options), ()

        nodes_and_input_options = tuple(
            (node, *split_input_options(node)) for node in chain.nodes
        )
        published = frozenset(chain.published)

        async def run_chain(accumulated_results, out_futures) -> _NodeToResults:
            chain_results: _NodeToResults = {}
            # Chain-internal results shadow the global mapping, so fan-out
            # within the chain is a plain dict lookup instead of a Future.
            lookup = collections.ChainMap(chain_results, accumulated_results)

            def concrete_value(source):
                value = chain_results.get(source, CG_NO_RESULT)
                if value is not CG_NO_RESULT:
                    return value
                if source in chain_set:
                    # Already ran and skipped; its published Future (if any)
                    # must not be read as a value.
                    raise KeyError(source)
                value = accumulated_results.get(source, CG_NO_RESULT)
                if value is CG_NO_RESULT:
                    raise KeyError(source)
                return value

            try:
                for node, concrete_options, remaining_options in (
                    nodes_and_input_options
                ):
                    args_kwargs = None
                    for args_spec, kwargs_spec in concrete_options:
                        try:
                            args_kwargs = (
                                tuple(concrete_value(arg) for arg in args_spec),
                                {
                                    key: concrete_value(value)
                                    for key, value in kwargs_spec.items()
                                },
                            )
                            break
                        except KeyError:
                            continue
                    if args_kwargs is None and remaining_options:
                        args_kwargs = await node_to_input_async(
                            lookup, remaining_options
                        )
                    if args_kwargs is None:
                        # Same as a per-node Task raising _DepNotFoundError: no
                        # result, consumers fall back to lower-priority options
                        # or skip. Resolving immediately (not at chain end) is
                        # what lets a consumer that reconverges into this chain
                        # through an async node proceed while the chain is
                        # still running.
                        if node in out_futures:
                            out_futures[node].set_exception(_DepNotFoundError())
                        continue
                    args, kwargs = args_kwargs
                    before = time.perf_counter()
                    try:
                        result = node.func(*args, **kwargs)
                    except handled as exception:
                        if node in out_futures:
                            out_futures[node].set_exception(exception)
                        continue
                    except Exception as exception:
                        if node in out_futures:
                            out_futures[node].set_exception(exception)
                        raise
                    single_node_side_effect(node, result)
                    if inspect.isawaitable(result):
                        raise Exception(
                            f"{node} returned an awaitable result but is not an async function"
                        )
                    _profile(node, before)
                    chain_results[node] = result
                    if node in out_futures:
                        out_futures[node].set_result(result)
                return chain_results
            finally:
                # After an unhandled exception the remaining nodes never ran;
                # resolve their futures so consumers and the final gather don't
                # wait forever.
                for future in out_futures.values():
                    if not future.done():
                        future.set_exception(_DepNotFoundError())
                # We delete the reference to the global mapping to avoid circular reference (task->exception->traceback->mapping) and improve memory performance
                del accumulated_results, lookup

        def schedule_chain(accumulated_results: dict, _chain) -> Awaitable:
            out_futures = {
                node: asyncio.get_running_loop().create_future() for node in published
            }
            accumulated_results.update(out_futures)
            return asyncio.create_task(run_chain(accumulated_results, out_futures))

        return schedule_chain

    node_to_consumers = collections.defaultdict(set)
    for edge in edges:
        for source in base_types.edge_sources(edge):
            node_to_consumers[source].add(edge.destination)
    node_to_chain = {}
    chains = _sync_chains(edges, sync_and_downstream)
    for chain_nodes in chains:
        chain_set = frozenset(chain_nodes)
        chain = _FusedSyncChain(
            chain_nodes,
            tuple(
                node
                for node in chain_nodes
                if node_to_consumers[node] - chain_set
            ),
        )
        for node in chain_nodes:
            node_to_chain[node] = chain
    logging.debug(
        f"sync chain fusion: fused {len(node_to_chain)} of {len(topological_sorted_nodes)} nodes into {len(chains)} chains"
    )
    units = []
    for node in topological_sorted_nodes:
        chain = node_to_chain.get(node)
        if chain is None:
            units.append((node, get_executor(node)))
        elif node == chain.nodes[0]:
            # The whole chain is scheduled as one unit at its head's position;
            # the other chain members are dropped from the walk.
            units.append((chain, make_chain_executor(chain)))
    return tuple(units)


async def _run_graph_async(inputs, handled_exceptions, execution_units):
    node_to_task_or_result = inputs.copy()
    unhandled_exception = None
    try:
        for node_executor in execution_units:
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
            for (node, node_result) in zip(
                async_results[0],
                await asyncio.gather(*async_results[1], return_exceptions=True),
            ):
                task_e = node_to_task_or_result[node].exception()
                if not task_e:
                    if type(node) is _FusedSyncChain:
                        # A fused chain's task returns the results of all its
                        # nodes (including unpublished ones), keyed by node.
                        all_results.update(node_result)
                    else:
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
