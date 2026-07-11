"""Authoring-time COLORING composers: declare which nodes belong to which color
while you BUILD the graph, and derive the runner's `run.NodeActivation` from it --
instead of reconstructing colors from a post-assembly graph's shape (`_core_reach`).

The runner already speaks coloring (`run.NodeActivation`, `run.ChangeActiveColors`,
`run.to_callable_with_coloring`); this module produces that spec.

Colors are stamped as TAGS on the func's `__dict__`, so they ride through
`duplicate_function` for free (functools.wraps copies `__dict__`) and survive any
amount of duplication -- provenance by tag, not by shape. `add_colors(COLORS,
subgraph, empty=...)` stamps `func.__cg_origin__` (+ an optional `__cg_empty__`),
`observer` stamps `__cg_observer__`, and `_pin_core_func` stamps the
absorbing `_CORE` marker. `build_node_activation_from_edges` then recovers the whole
`run.NodeActivation` from the final graph's tags: the SINGLE-COLOR RULE (>=2 colors =
shared = always-on), a tolerance-aware must-run closure, and combiner-aware boundary
defaults from the `empty` tags. `latch` makes a value survive the turn its producer
prunes (pinning its own machinery core).

The engine stays domain-agnostic: colors are opaque tokens; nothing here knows
skills / routes / effects.
"""
from __future__ import annotations

import collections
from typing import Any, Callable, Dict, FrozenSet, Mapping, Optional

from computation_graph import base_types, composers, graph, run

_ORIGIN_ATTR = "__cg_origin__"
_EMPTY_ATTR = "__cg_empty__"
_OBSERVER_ATTR = "__cg_observer__"
_UNSET = object()


# --------------------------------------------------------------------------- #
# Node selection: sources are caller-seeded and terminals are sinks; neither is
# ever skipped, so neither is colorable.
# --------------------------------------------------------------------------- #
def _edges(graph_or_edges) -> FrozenSet[base_types.ComputationEdge]:
    # Every public entry point accepts either a `GraphType` or a bare edge
    # collection; internals below always work on edges.
    return (
        graph_or_edges.edges
        if isinstance(graph_or_edges, base_types.GraphType)
        else graph_or_edges
    )


def _is_source(node: base_types.ComputationNode) -> bool:
    return getattr(node.func, "__name__", "").startswith("source:")


def _colorable_nodes(graph_or_edges) -> FrozenSet[base_types.ComputationNode]:
    edges = _edges(graph_or_edges)
    return frozenset(
        n for n in graph.get_all_nodes(edges) if not n.is_terminal and not _is_source(n)
    )


def _leaves(edges) -> FrozenSet[base_types.ComputationNode]:
    all_sources = frozenset(
        source for edge in edges for source in base_types.edge_sources(edge)
    )
    return graph.get_all_nodes(edges) - all_sources


def _sinks(graph_or_edges) -> FrozenSet[base_types.ComputationNode]:
    edges = _edges(graph_or_edges)
    return frozenset(
        n for n in _leaves(edges) if not n.is_terminal and not _is_source(n)
    )


# --------------------------------------------------------------------------- #
# The color tag rides the func's `__dict__`, so it survives `duplicate_function`
# (functools.wraps copies `__dict__`) and is recoverable from the assembled graph.
# The tag is a FROZENSET of opaque color tokens (colors UNION when more than one
# author tags the same func -- a building block shared by reference among skills A
# and B ends up {A, B}), OR the absorbing sentinel `_CORE` meaning "never color this"
# (a core/shared func, pinned at its authoring site, stays uncolored even when a
# skill's sweep later passes over a duplicated copy of it). Tags are NEVER cleared per
# build: colors ACCUMULATE, so a func shared across bots ends up multi-colored and
# always-runs (>=2 colors) -- provenance by tag, immune to a foreign single color.
# --------------------------------------------------------------------------- #
_CORE = "\x00cg-core"  # absorbing marker; not a usable color token


def _tag_func(func, attr: str, value) -> None:
    # The one guarded stamp: write `attr = value` onto the func's __dict__ so it rides
    # `duplicate_function` (functools.wraps copies __dict__). No-op on an un-taggable
    # builtin / C callable -> it stays tag-less, which the readers treat as core.
    try:
        setattr(func, attr, value)
    except (AttributeError, TypeError):
        pass


def add_colors(
    colors: FrozenSet, subgraph: base_types.GraphType, *, empty: Any = _UNSET
) -> base_types.GraphType:
    """Union `colors` into every colorable node's tag. A func already pinned core
    is left untouched (core is absorbing). Returns the graph unchanged (the tag
    rides each func, so it survives duplication).

    `empty` (optional) is the subgraph's typed-empty output: it is stamped on the
    sink func(s) so a pruned skill contributes "nothing" at an INTOLERANT frontier
    (an aggregation / make_and that needs all its args). Like the color tag it rides
    `func.__dict__` through duplication; `build_node_activation_from_edges` reads it
    back at each intolerant colored->core frontier. TOLERANT frontiers (make_or /
    make_first sinks, and terminals -- which the runner merges via make_or) never
    consult it, so only the genuinely intolerant frontier needs an `empty`."""
    for node in _colorable_nodes(subgraph):
        current = getattr(node.func, _ORIGIN_ATTR, None)
        if current == _CORE:
            continue
        _tag_func(node.func, _ORIGIN_ATTR, (current or frozenset()) | colors)
    if empty is not _UNSET:
        tag_empty(empty, subgraph)
    return subgraph


def tag_empty(empty: Any, subgraph: base_types.GraphType) -> base_types.GraphType:
    """Stamp `empty` (a typed-empty value) on every sink func of `subgraph` so an
    intolerant frontier consuming that sink can be satisfied without it when the
    producer is pruned. Standalone analogue of `add_colors(..., empty=...)` for
    when the empty is attached independently of coloring."""
    for sink in _sinks(subgraph):
        _tag_func(sink.func, _EMPTY_ATTR, empty)
    return subgraph


def _read_empties(
    graph_or_edges,
) -> Dict[base_types.ComputationNode, Any]:
    """node -> its typed-empty, for every node whose func carries an `empty` tag."""
    out: Dict[base_types.ComputationNode, Any] = {}
    for node in graph.get_all_nodes(_edges(graph_or_edges)):
        value = getattr(node.func, _EMPTY_ATTR, _UNSET)
        if value is not _UNSET:
            out[node] = value
    return out


def _pin_core_func(func):
    # Pin a single callable CORE (absorbing -- a later color sweep leaves it alone).
    # The CORE tag rides func.__dict__ through duplicate_function, so copies stay core
    # too. Returns the func. No-op on un-taggable (builtin/C) callables.
    _tag_func(func, _ORIGIN_ATTR, _CORE)
    return func


def pin_core(subgraph_or_callable):
    """Pin every colorable node of a subgraph -- or a single bare callable -- CORE
    (absorbing): a later `add_colors` sweep passing over these funcs -- or over any
    DUPLICATE of them -- leaves them uncolored, so they always run. Use at the
    AUTHORING SITE of shared machinery that skills later compose into their own
    channels (routing state, shared event chains, variable-exposure derivations):
    without the pin, each per-skill duplicate ends up inside exactly one skill's
    closure -> single color -> pruned on every other skill's turn, starving core
    flows. The tag rides func.__dict__ through duplication. Returns the argument
    unchanged."""
    if not isinstance(
        subgraph_or_callable, (base_types.GraphType, tuple, list, set, frozenset)
    ):
        _pin_core_func(subgraph_or_callable)
        return subgraph_or_callable
    for node in _colorable_nodes(subgraph_or_callable):
        _pin_core_func(node.func)
    return subgraph_or_callable


def _read_colors(
    graph_or_edges,
) -> Mapping[base_types.ComputationNode, FrozenSet]:
    """node -> its color set, for every colored node (core / untagged omitted)."""
    out = {}
    for node in graph.get_all_nodes(_edges(graph_or_edges)):
        tag = getattr(node.func, _ORIGIN_ATTR, None)
        if tag and tag != _CORE:
            out[node] = tag
    return out


# --------------------------------------------------------------------------- #
# Observers: nodes that compare-or-accumulate across turns (lag / changed / ever /
# accumulate, and any domain combinator that does the same). `observer` tags them
# (via `_mark_observer`) and also pins their future-state machinery core. NOTE:
# `build_node_activation_from_edges` does NOT exempt observers -- a skill-private
# observer prunes WITH its skill (it feeds the tolerant EVENT path so an inactive
# skill emits no event, and its accumulated state survives the pruned turn via the
# reducer's {**prev} latch on its future self-edge). Forcing observer cones always-on
# was measured to drag in ~the whole graph. The mark therefore drives diagnostics and
# is available for a caller that wants to build an explicit must-tick latch on an
# observer's input; the machinery pin is defensive (a same-color when_memory_unavailable
# wouldn't create a frontier anyway).
# --------------------------------------------------------------------------- #
def _mark_observer(func):
    # Mark a node func as a must-tick observer (rides `func.__dict__` through
    # duplication). Returns the func. Internal: applied by `observer` below.
    _tag_func(func, _OBSERVER_ATTR, True)
    return func


def observer(result: base_types.GraphType, inner: Callable) -> base_types.GraphType:
    """Declare `inner` (the across-turns node of the observer combinator `result`) an
    observer: `_mark_observer` it and pin its own future-state machinery CORE (the
    `when_memory_unavailable` default node its future self-edge creates). See the note
    above -- observers are NOT exempted by the activation builder; the mark drives
    diagnostics / an optional caller-built must-tick latch, and the pin is defensive.
    Use for any domain combinator that compares-or-accumulates across turns."""
    _mark_observer(inner)
    for node in graph.get_all_nodes(_edges(result)):
        if getattr(node.func, "__name__", "") == "when_memory_unavailable":
            _pin_core_func(node.func)
    return result


# --------------------------------------------------------------------------- #
# The latch: make a value SURVIVE the turn its producer is pruned (the producing
# color inactive) without a boundary default. Generalizes nlu's var-bus
# `_LatchConsumeVariable`. The producer feeds a tolerant frontier (a make_or
# aggregate / a terminal) so a pruned producer is filtered to "absent"; the latch
# then returns the value's own PREVIOUS (a future self-edge) instead of the absent
# default, so it persists across the pruned turn.
# --------------------------------------------------------------------------- #
class _Pruned:
    """Sentinel the latch guard yields when `current`'s producer was PRUNED this
    turn (produced nothing). Distinct from any domain value -- in particular from a
    domain-level "unknown" a producer that RAN can legitimately return. Private:
    `carry_forward` translates it before any consumer sees it."""

    def __repr__(self):
        return "_Pruned()"


_PRUNED = _Pruned()


def latch(
    current: base_types.CallableOrNodeOrGraph,
    *,
    present: Optional[Callable[[Any], bool]] = None,
    default: Any,
) -> base_types.GraphType:
    """Carry `current` forward across turns it is absent. A PRUNED producer (its
    color inactive this turn) always falls back to `previous`. `present(value)`, if
    given, additionally treats this turn's PRODUCED value as absent when false --
    callers whose produced values encode absence (e.g. a consume terminal's
    all-exposers-pruned sentinel) use it; omit it to latch on pruning ONLY, letting
    a produced domain-unknown pass through exactly as the always-run baseline would
    deliver it (latching over it would resurrect a stale value). `default` is the
    value before any turn has produced one. The latch node + the
    `when_memory_unavailable` default node its future self-edge creates are pinned
    CORE (they are shared carry-forward machinery that must run every turn) so the
    per-color sweep can't mis-color them; `current`'s own subgraph stays
    colored/prunable. Returns a graph whose sink is the latch -- compose consumers
    onto it."""

    def carry_forward(current, previous):
        if current is _PRUNED:
            return previous
        return previous if present is not None and not present(current) else current

    def latched(value):  # pass-through: a future-edge-FREE sink, so consumers and
        return value  # `sink_excluding_terminals` resolve cleanly (carry_forward's
        # own future self-edge would otherwise leave the graph with no plain leaf).

    def latch_default():
        return _PRUNED

    # `current` enters through a TOLERANT frontier (make_first + a sentinel
    # fallback): a pruned producer yields `_PRUNED` (-> previous) instead of
    # STARVING the pinned carry_forward. Without this guard, a colored `current` at
    # the pinned-intolerant frontier is caught by the must-run closure, which
    # forces its whole input cone always-on -- pruning collapses (measured:
    # bon_secours prunable 50k -> 26k, greeting 257ms -> 11s). The sentinel (not
    # `default`) is what lets pruned be distinguished from ran-and-returned-unknown.
    # A terminal `current` (the var-bus consume) is already tolerant; the guard is
    # harmless there.
    guarded_current = composers.make_first(current, latch_default)
    self_edge = composers.compose_left_future(
        carry_forward, carry_forward, "previous", default
    )
    latched_graph = composers.compose_left_unary(carry_forward, latched)
    g = graph.merge_graphs(
        composers.compose_dict(carry_forward, {"current": guarded_current}),
        self_edge,
        latched_graph,
        sink_node_or_graph=latched_graph,
    )
    # Pin the latch's own machinery CORE (carry_forward + the when_memory_unavailable
    # default node + the guard's first_sink/fallback + the pass-through) -- shared
    # carry-forward infra that must run every turn; `current`'s subgraph stays
    # colored/prunable.
    for node in _colorable_nodes(self_edge):
        _pin_core_func(node.func)
    _pin_core_func(latch_default)
    _pin_core_func(latched)
    current_nodes = (
        _colorable_nodes(current)
        if isinstance(current, (base_types.GraphType, tuple, frozenset, set))
        else frozenset()
    )
    for node in _colorable_nodes(guarded_current) - current_nodes:
        if callable(current) and node.func is current:
            continue
        _pin_core_func(node.func)
    return g


# --------------------------------------------------------------------------- #
# Derive the runner spec.
# --------------------------------------------------------------------------- #
class BoundaryDefaultRequired(Exception):
    def __init__(self, message, missing=()):
        super().__init__(message)
        # [(producer, consumer)] -- the intolerant frontiers lacking a typed-empty,
        # so callers/diagnostics can categorize them without parsing the message.
        self.missing = tuple(missing)


def _is_tolerant_consumer(node: base_types.ComputationNode) -> bool:
    # A consumer tolerates a pruned input iff a missing arg does not collapse it:
    #   * make_or / make_first / make_optional funnel each input into a `first_sink`
    #     node, which tolerates a missing input by construction;
    #   * a TERMINAL -- the runner merges all edges into a terminal via `make_or`
    #     (`run._merge_edges_pointing_to_terminals`), so a pruned producer is filtered
    #     and the terminal's aggregate just shrinks (this is what lets a var-bus
    #     exposer prune with no default, latched downstream).
    # `duplicate_function` prefixes __name__ with "duplicate of ", so strip it --
    # a DUPLICATED first_sink is exactly as tolerant as the original (missing this
    # misclassified duplicated sinks as intolerant and closure-forced their whole
    # producer cones always-on).
    name = getattr(node.func, "__name__", "")
    while name.startswith("duplicate of "):
        name = name[len("duplicate of ") :]
    return node.is_terminal or name == "first_sink"


def _needs_default(producer_colors: FrozenSet, consumer_colors) -> bool:
    # A default is needed iff some active set runs the CONSUMER but skips the PRODUCER:
    #   * uncolored (core) consumer always runs                -> needed
    #   * colored consumer with a color the producer lacks     -> needed
    if consumer_colors is None:
        return True
    return bool(consumer_colors - producer_colors)


# --------------------------------------------------------------------------- #
# Derive the runner spec straight from the assembled graph's TAGS (colors / empties /
# observer marks): the author tags survive duplication on the funcs, so the whole
# activation is recovered from `edges` alone -- no in-memory ColoredGraph needed.
# --------------------------------------------------------------------------- #
def _intolerant_frontiers(edges, node_to_colors, empties):
    """The colored->(core / other-color) frontiers whose consumer is INTOLERANT (not
    a first_sink, not a terminal). Yields (producer, consumer, producer_is_defaultable).
    A producer prunes when its (single) color is inactive; the consumer still runs iff
    it does not share that color (`_needs_default`) -- that is the frontier the runner
    must handle, by the producer's `empty` tag if it has one, else by the producer
    being forced always-on."""
    for edge in edges:
        consumer = edge.destination
        if _is_tolerant_consumer(consumer):
            continue
        consumer_colors = node_to_colors.get(consumer)
        for producer in base_types.edge_sources(edge):
            producer_colors = node_to_colors.get(producer)
            if producer_colors is None:
                continue  # core / shared / observer producer -> never pruned
            if not _needs_default(producer_colors, consumer_colors):
                continue  # same color -> prune together
            yield producer, consumer, producer in empties


def _must_run_closure(edges, seeds):
    """`seeds` plus their intolerant input cone: backward from each seed, a node's
    producers are pulled in only while the node is an INTOLERANT consumer (needs all
    its args). Stop at tolerant consumers (first_sink / terminals) -- their inputs may
    prune, so the cone ends there. Tolerance-aware, tag-driven analogue of a data-flow
    core-reach: everything reached must run every turn."""
    incoming: Dict = collections.defaultdict(list)
    for edge in edges:
        for producer in base_types.edge_sources(edge):
            incoming[edge.destination].append(producer)
    seen: set = set()
    stack = list(seeds)
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        if _is_tolerant_consumer(node):
            continue  # tolerant: its inputs may prune, don't pull them in
        stack.extend(incoming.get(node, ()))
    return seen


def build_node_activation_from_edges(
    graph_or_edges, *, strict: bool = False
) -> run.NodeActivation:
    """Derive `run.NodeActivation` from an assembled, possibly-duplicated graph using
    only the author tags carried on the node funcs:

      * COLORS  (`_read_colors`)  -- the per-node color sets.
      * SINGLE-COLOR RULE -- only nodes with EXACTLY ONE color are colored (prunable);
        a node with >=2 colors is SHARED and always-runs (an authoring-time proxy for
        "shared -> feeds core"), an untagged node is CORE.
      * MUST-RUN CLOSURE -- every colored producer at an INTOLERANT frontier with NO
        `empty` tag, PLUS its intolerant input cone, is forced always-on: an always-on
        intolerant consumer needs its REAL inputs (it has no default to substitute).
        This is the tolerance-aware, tag-driven reduction of a data-flow core-reach --
        fully generic (CG owns the tolerance rule). An `empty` tag opts a frontier OUT
        of the closure (the producer prunes and the runner substitutes the empty) --
        that is how the caller carves prunable regions (e.g. the effect aggregation)
        out of always-on core.
      * BOUNDARY DEFAULTS -- the surviving colored producers at intolerant frontiers,
        each with its `empty` tag.

    OBSERVERS ARE NOT SPECIAL-CASED. A skill-private observer (`source_changed` / `ever`
    / `lag` / `accumulate` colored a single skill) prunes WITH its skill: it feeds the
    tolerant EVENT path (no event for an inactive skill -- correct), and its accumulated
    state survives the pruned turn for free via the reducer's `{**prev}` latch on its
    future self-edge (it resumes when the skill reactivates, having merely paused while
    its skill-private input wasn't being produced). An observer that DOES feed an
    intolerant core consumer is caught by the generic closure above, like any producer.
    (Forcing observer cones always-on was measured to drag in ~the whole graph, since
    observers transitively depend on nearly everything -- it defeats pruning entirely.)
    The `__cg_observer__` mark (stamped by `observer`) is available to a caller that
    wants to build an explicit must-tick latch on an observer's input.

    `strict=True` instead RAISES `BoundaryDefaultRequired` listing the untagged
    intolerant frontiers (an audit of what falls back to always-on) rather than forcing
    them core."""
    edges = _edges(graph_or_edges)
    colors = _read_colors(edges)
    node_to_colors = {n: c for n, c in colors.items() if len(c) == 1}
    empties = _read_empties(edges)

    untagged = [
        (p, c) for p, c, defaultable in _intolerant_frontiers(edges, node_to_colors, empties)
        if not defaultable
    ]
    if strict and untagged:
        raise BoundaryDefaultRequired(
            "intolerant colored->core frontier without a typed-empty "
            "(tag it with add_colors(..., empty=...) / tag_empty):\n"
            + "\n".join(
                f"  {p.name} {sorted(node_to_colors[p])} --> {c.name}"
                for p, c in untagged
            ),
            missing=untagged,
        )
    # Force always-on: untagged intolerant-frontier producers + their intolerant cone.
    for node in _must_run_closure(edges, {p for p, _ in untagged}):
        node_to_colors.pop(node, None)

    boundary_defaults: Dict[base_types.ComputationNode, Any] = {}
    for producer, _consumer, defaultable in _intolerant_frontiers(
        edges, node_to_colors, empties
    ):
        if defaultable:  # producer survived the closure (had an empty) -> prune w/ it
            boundary_defaults[producer] = empties[producer]
    return run.NodeActivation(node_to_colors, boundary_defaults)
