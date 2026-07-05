"""Sync-chain fusion: run each maximal linear chain of synchronous nodes that
are downstream of async nodes inside a single asyncio Task, instead of one
Task per node.

Rationale: in an async graph, every sync node downstream of an async node is
scheduled as its own asyncio Task (see run.py's await_deps_and_apply), costing
tens of microseconds of Task machinery for functions whose body is often
sub-microsecond. A linear chain (in-degree = out-degree = 1 within the
subgraph induced on such nodes) is the zero-concurrency-loss unit of fusion:
every chain member depends on its predecessor, so executing the chain
sequentially in one Task cannot delay anything that could have run earlier.

Semantics are preserved by construction:
- Each chain node resolves its inputs with the same option/priority logic as
  unfused nodes (the run.py closures arrive via FusionContext); external async
  inputs are awaited lazily, exactly when the needing node is reached.
- A chain node consumed outside its chain gets a Future in the results
  mapping, resolved the moment the node computes or skips ("eager
  publication"). This is what makes reconvergence through an async node
  (b -> c(async) -> d alongside a fused chain b -> d) deadlock-free: c awaits
  b's already-published Future while the chain awaits c.
- Skips (_DepNotFoundError / SkipComputationError / handled exceptions) mean
  the node's value is absent and its Future (if any) carries the exception, so
  consumers fall back to lower-priority options or skip, as without fusion.

Off by default; enabled at graph-compilation time by setting the
COMPUTATION_GRAPH_SYNC_FUSION environment variable (to anything but
"0"/"false"). With COMPUTATION_GRAPH_DEBUG also set, every evaluation runs
both fused and unfused and asserts the same nodes were computed.
"""
import asyncio
import collections
import dataclasses
import inspect
import logging
import os
import time
from typing import Callable, Set, Tuple, Type

import gamla

from computation_graph import base_types, graph

COMPUTATION_GRAPH_SYNC_FUSION_ENV_KEY = "COMPUTATION_GRAPH_SYNC_FUSION"

_NO_RESULT = object()


def enabled() -> bool:
    return os.getenv(COMPUTATION_GRAPH_SYNC_FUSION_ENV_KEY, "0") not in (
        "0",
        "false",
        "False",
        "",
    )


class FusedSyncChain:
    """A maximal linear run of synchronous nodes downstream of async nodes.

    `published` are the chain nodes whose result is consumed outside the
    chain; each gets a Future in the results mapping so external consumers can
    await it.
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


@dataclasses.dataclass(frozen=True)
class FusionContext:
    """The compile-time internals of run.py that chain executors reuse, so
    fused nodes resolve inputs and report side effects exactly like unfused
    ones."""

    edges: base_types.GraphType
    handled_exceptions: Tuple[Type[Exception], ...]
    single_node_side_effect: Callable
    node_to_input_async: Callable
    node_to_input_spec_options: Callable
    sync_and_downstream: Set[base_types.ComputationNode]
    sync_not_downstream: Set[base_types.ComputationNode]
    dep_not_found_error: Type[Exception]
    profile: Callable


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


def _make_chain_executor(
    chain: FusedSyncChain,
    statically_concrete: Set[base_types.ComputationNode],
    context: FusionContext,
):
    chain_set = frozenset(chain.nodes)
    node_to_input_async = context.node_to_input_async
    single_node_side_effect = context.single_node_side_effect
    profile = context.profile
    dep_not_found_error = context.dep_not_found_error
    handled = (base_types.SkipComputationError, *context.handled_exceptions)

    def split_input_options(node):
        """Prefix of priority-ordered options resolvable with plain dict
        lookups; from the first option that may reference a Future on, the
        rest is handled by node_to_input_async (which resolves concrete values
        too, so order is preserved)."""
        options = context.node_to_input_spec_options(node)
        concrete_options: list = []
        for index, option in enumerate(options):
            args_spec, kwargs_spec = option
            if all(
                source in statically_concrete or source in chain_set
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

    async def run_chain(accumulated_results, out_futures):
        chain_results: dict = {}
        # Chain-internal results shadow the global mapping, so fan-out within
        # the chain is a plain dict lookup instead of a Future.
        lookup = collections.ChainMap(chain_results, accumulated_results)

        def concrete_value(source):
            value = chain_results.get(source, _NO_RESULT)
            if value is not _NO_RESULT:
                return value
            if source in chain_set:
                # Already ran and skipped; its published Future (if any) must
                # not be read as a value.
                raise KeyError(source)
            value = accumulated_results.get(source, _NO_RESULT)
            if value is _NO_RESULT:
                raise KeyError(source)
            return value

        try:
            for node, concrete_options, remaining_options in nodes_and_input_options:
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
                    args_kwargs = await node_to_input_async(lookup, remaining_options)
                if args_kwargs is None:
                    # Same as a per-node Task raising _DepNotFoundError: no
                    # result, consumers fall back to lower-priority options or
                    # skip. Resolving immediately (not at chain end) is what
                    # lets a consumer that reconverges into this chain through
                    # an async node proceed while the chain is still running.
                    if node in out_futures:
                        out_futures[node].set_exception(dep_not_found_error())
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
                profile(node, before)
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
                    future.set_exception(dep_not_found_error())
            # We delete the reference to the global mapping to avoid circular reference (task->exception->traceback->mapping) and improve memory performance
            del accumulated_results, lookup

    def schedule_chain(accumulated_results: dict, _chain):
        out_futures = {
            node: asyncio.get_running_loop().create_future() for node in published
        }
        accumulated_results.update(out_futures)
        return asyncio.create_task(run_chain(accumulated_results, out_futures))

    return schedule_chain


def fuse(node_executor_pairs: Tuple, context: FusionContext) -> Tuple:
    """Replace every fusable chain in the (node, executor) execution order
    with a single (FusedSyncChain, executor) unit at its head's position; all
    other units are passed through untouched."""
    chains = _sync_chains(context.edges, context.sync_and_downstream)
    if not chains:
        return node_executor_pairs
    topological_sorted_nodes = tuple(pair[0] for pair in node_executor_pairs)
    # Nodes whose value in the results mapping is never a Future: inputs
    # (placeholders and sources are removed from the executed topological
    # order) and nodes computed by direct call in the walk. Chain members are
    # added per chain: by execution order they are plain values (or absent) by
    # the time a later chain node reads them.
    statically_concrete = context.sync_not_downstream | (
        graph.get_all_nodes(context.edges) - set(topological_sorted_nodes)
    )
    node_to_consumers = collections.defaultdict(set)
    for edge in context.edges:
        for source in base_types.edge_sources(edge):
            node_to_consumers[source].add(edge.destination)
    node_to_chain = {}
    for chain_nodes in chains:
        chain_set = frozenset(chain_nodes)
        chain = FusedSyncChain(
            chain_nodes,
            tuple(node for node in chain_nodes if node_to_consumers[node] - chain_set),
        )
        for node in chain_nodes:
            node_to_chain[node] = chain
    logging.debug(
        f"sync chain fusion: fused {len(node_to_chain)} of {len(topological_sorted_nodes)} nodes into {len(chains)} chains"
    )
    units = []
    for node, executor in node_executor_pairs:
        chain = node_to_chain.get(node)
        if chain is None:
            units.append((node, executor))
        elif node == chain.nodes[0]:
            units.append(
                (chain, _make_chain_executor(chain, statically_concrete, context))
            )
    return tuple(units)


def with_debug_equivalence_check(fused_runner: Callable, unfused_runner: Callable):
    """Debug-mode wrapper: run the graph without fusion as well and compare.
    Note this runs every node twice per evaluation."""

    async def final_runner(sources_to_values):
        fused_results = await fused_runner(sources_to_values)
        _assert_equivalence(fused_results, await unfused_runner(sources_to_values))
        return fused_results

    return final_runner


def _assert_equivalence(fused, unfused):
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
