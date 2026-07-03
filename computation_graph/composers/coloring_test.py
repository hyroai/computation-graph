"""Tests for the authoring-time coloring composers (computation_graph.composers.coloring).

The color / empty / observer tags ride `func.__dict__` through duplication, and
`build_node_activation_from_edges` recovers the whole `run.NodeActivation` from those
tags: the single-color rule, the tolerance-aware must-run closure, and combiner-aware
boundary defaults. `latch` carries a value across the turn its producer prunes.
"""
import gamla
import pytest

from computation_graph import base_types, composers, graph, run
from computation_graph.composers import coloring, duplication, memory


def test_color_tag_survives_duplication():
    def a(x):
        return x

    def b(y):
        return y

    x = graph.make_source()
    sub = composers.compose_dict(b, {"y": composers.compose_left_unary(x, a)})
    coloring.add_colors(frozenset({"skill-X"}), sub)

    duplicated = duplication.duplicate_graph(sub)
    colors = coloring._read_colors(duplicated)

    colorable = [
        n for n in graph.get_all_nodes(duplicated)
        if not n.is_terminal and not getattr(n.func, "__name__", "").startswith("source:")
    ]
    assert colorable, "expected colorable nodes after duplication"
    assert all(colors.get(n) == frozenset({"skill-X"}) for n in colorable)
    # genuinely new nodes (duplication changed identity)
    originals = {graph.make_computation_node(a), graph.make_computation_node(b)}
    assert not (originals & set(colorable))


def test_add_colors_unions_shared_by_reference():
    def shared(x):
        return x

    def private(y):
        return y

    x = graph.make_source()
    sub_a = composers.compose_left_unary(x, shared)
    sub_b = composers.compose_left_unary(shared, private)
    coloring.add_colors(frozenset({"A"}), sub_a)
    coloring.add_colors(frozenset({"B"}), sub_b)  # `shared` is in both -> union

    colors = coloring._read_colors(sub_b)
    assert colors[graph.make_computation_node(shared)] == frozenset({"A", "B"})
    assert colors[graph.make_computation_node(private)] == frozenset({"B"})


def test_pin_core_func_is_absorbing():
    def shared(x):
        return x

    def private(y):
        return y

    coloring._pin_core_func(shared)  # the shared building block pins itself core first
    skill = composers.compose_left_unary(shared, private)
    coloring.add_colors(frozenset({"skill-X"}), skill)  # sweep can't color `shared`

    colors = coloring._read_colors(skill)
    assert graph.make_computation_node(shared) not in colors  # stayed core
    assert colors[graph.make_computation_node(private)] == frozenset({"skill-X"})


# --------------------------------------------------------------------------- #
# Tier 2: build_node_activation_from_edges (tags -> activation)
# --------------------------------------------------------------------------- #
def _tagged_make_and(calls, *, with_empties):
    def skill_a(x):
        calls.append("A")
        return ("A", x)

    def skill_b(x):
        calls.append("B")
        return ("B", x)

    def combine(a, b):
        return (a, b)

    x = graph.make_source()
    ga = composers.compose_left_future(x, skill_a, None, None)
    gb = composers.compose_left_future(x, skill_b, None, None)
    empty_a = {"empty": ("A", "EMPTY")} if with_empties else {}
    empty_b = {"empty": ("B", "EMPTY")} if with_empties else {}
    coloring.add_colors(frozenset({"A"}), ga, **empty_a)
    coloring.add_colors(frozenset({"B"}), gb, **empty_b)
    core = composers.compose_dict(combine, {"a": skill_a, "b": skill_b})
    g = base_types.merge_graphs(ga, gb, core)
    return x, g, skill_a, skill_b, combine


def test_empty_tag_drives_intolerant_frontier_default():
    calls = []
    x, g, skill_a, skill_b, combine = _tagged_make_and(calls, with_empties=True)
    activation = coloring.build_node_activation_from_edges(g)

    n_a, n_b = graph.make_computation_node(skill_a), graph.make_computation_node(skill_b)
    assert activation.boundary_defaults == {n_a: ("A", "EMPTY"), n_b: ("B", "EMPTY")}
    assert activation.node_to_colors[n_a] == frozenset({"A"})

    f = run.to_callable_with_coloring(g, frozenset())
    out = f({}, {x: 5}, frozenset({"A"}))
    assert calls == ["A"]  # only A ran; B's tagged empty flowed into combine
    assert out[graph.make_computation_node(combine)] == (("A", 5), ("B", "EMPTY"))


def test_missing_empty_at_intolerant_frontier_raises_in_strict_mode():
    x, g, skill_a, skill_b, _ = _tagged_make_and([], with_empties=False)
    with pytest.raises(coloring.BoundaryDefaultRequired) as e:
        coloring.build_node_activation_from_edges(g, strict=True)
    assert "skill_a" in str(e.value) and "skill_b" in str(e.value)


def test_missing_empty_forces_producer_always_on_by_default():
    # Graceful (default): an untagged intolerant frontier forces its producer
    # always-on (core) rather than raising -- safe, never starves the consumer.
    x, g, skill_a, skill_b, _ = _tagged_make_and([], with_empties=False)
    activation = coloring.build_node_activation_from_edges(g)
    n_a, n_b = graph.make_computation_node(skill_a), graph.make_computation_node(skill_b)
    assert n_a not in activation.node_to_colors  # forced always-on
    assert n_b not in activation.node_to_colors
    assert activation.boundary_defaults == {}


def test_single_color_rule_shared_node_always_runs():
    def shared(x):
        return x

    x = graph.make_source()
    sub = composers.compose_left_unary(x, shared)
    coloring.add_colors(frozenset({"A"}), sub)
    coloring.add_colors(frozenset({"B"}), sub)  # shared -> {A, B}: shared, always-on

    activation = coloring.build_node_activation_from_edges(sub)
    # >=2 colors -> not prunable (shared infra runs every turn).
    assert graph.make_computation_node(shared) not in activation.node_to_colors


def test_observer_prunes_with_its_skill():
    def watched(v):
        return bool(v)

    coloring._pin_core_func(watched)  # the watched value is core (always-on)
    x = graph.make_source()
    obs = memory.ever(composers.compose_left_unary(x, watched))
    coloring.add_colors(frozenset({"A"}), obs)  # colors ever_inner A

    activation = coloring.build_node_activation_from_edges(obs)
    ever_node = next(
        n for n in graph.get_all_nodes(obs)
        if getattr(n.func, "__name__", "") == "ever_inner"
    )
    # `memory.ever` marked it an observer, but the builder does NOT special-case
    # observers: a skill-private observer prunes WITH its skill (forcing observer cones
    # always-on was measured to drag in ~the whole graph).
    assert getattr(ever_node.func, coloring._OBSERVER_ATTR, False)
    assert activation.node_to_colors[ever_node] == frozenset({"A"})


def test_latch_survives_pruned_producer():
    UNKNOWN = "UNKNOWN"
    WANT = ("B", "phone:555")
    calls = []

    def expose_a(utterance):  # skill A exposes the var. colored {A}
        calls.append("A")
        return "phone:555"

    def consume_var(val):  # the var terminal: first present exposer or absent
        return val[0] if len(val) > 0 else UNKNOWN

    def use_b(v):  # skill B reads the var. colored {B}
        calls.append("B")
        return ("B", v)

    u = graph.make_source()
    term = graph.make_terminal("VAR", consume_var)
    expose_graph = composers.compose_left_future(u, expose_a, None, None)
    coloring.add_colors(frozenset({"A"}), expose_graph)
    latched = coloring.latch(term, present=lambda v: v != UNKNOWN, default=UNKNOWN)
    use_graph = composers.compose_left_unary(latched, use_b)
    coloring.add_colors(frozenset({"B"}), use_graph)  # latch is pinned core -> only use_b colored
    g = base_types.merge_graphs(
        expose_graph,
        composers.compose_left_unary(expose_a, term),  # exposer -> terminal (tolerant)
        use_graph,
    )
    f = run.to_callable_with_coloring(g, frozenset())
    r1 = f({}, {u: "hi"}, frozenset({"A"}))  # turn 1: A active -> exposes the var
    r2 = f(r1, {u: "hi"}, frozenset({"B"}))  # turn 2: B active, A pruned -> latch holds it

    assert r2[graph.make_computation_node(use_b)] == WANT
    assert calls == ["A", "B"]  # A ran once (turn 1), B ran once (turn 2)
