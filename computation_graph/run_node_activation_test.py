"""Tests for the opt-in per-node skipping ("coloring") in run.to_callable.

The engine is domain-agnostic: it knows only COLORS (opaque tokens) and the
`run.ChangeActiveColors` event. The CALLER passes the initial active colors when it
starts the run (the reducer's `active_colors` argument; optional). Colored nodes
whose color is not active skip; with no initial color, only the no-color nodes run.
If a node then RETURNS `run.ChangeActiveColors(other)` the run STARTS OVER with the
new color set (nodes skipped on the first pass may now need to run). Validates:
  * a SKIPPED BOUNDARY node returns its typed-empty default (downstream make_and
    stays well-formed); a SKIPPED INTERIOR node (no default) prunes (absent);
  * the caller's initial color selects which skill runs (single pass);
  * a ChangeActiveColors event restarts the run to the newly-active color.
"""
import asyncio

import gamla

from computation_graph import base_types, composers, graph, run


def _to_callable(edges, activation):
    # Compile a graph with an explicit, hand-built NodeActivation (production derives
    # one from tags via `run.to_callable_with_coloring`; these tests drive the runner
    # skipping directly). Same null-side-effect path as `run.to_callable`, with the
    # activation forwarded as its trailing argument.
    return run.to_callable_with_side_effect(
        gamla.just(gamla.just(None)), edges, frozenset(), activation
    )


# --------------------------------------------------------------------------- #
# Sync: skill_a/skill_b colored {A}/{B}; combine is make_and over both. The
# caller passes the active color directly.
# --------------------------------------------------------------------------- #
def _build(calls):
    def skill_a(x):
        calls.append("A")
        return ("A", x)

    def skill_b(x):
        calls.append("B")
        return ("B", x)

    def combine(a, b):
        return (a, b)

    x_source = graph.make_source()
    g = base_types.merge_graphs(
        composers.compose_left_future(x_source, skill_a, None, None),
        composers.compose_left_future(x_source, skill_b, None, None),
        composers.compose_dict(combine, {"a": skill_a, "b": skill_b}),
    )
    nodes = {
        "a": graph.make_computation_node(skill_a),
        "b": graph.make_computation_node(skill_b),
        "combine": graph.make_computation_node(combine),
    }
    return g, x_source, nodes


def _activation(nodes):
    return run.NodeActivation(
        node_to_colors={nodes["a"]: frozenset({"A"}), nodes["b"]: frozenset({"B"})},
        boundary_defaults={nodes["a"]: ("A", "EMPTY"), nodes["b"]: ("B", "EMPTY")},
    )


def test_initial_color_runs_matching_skill():
    calls = []
    g, x_source, nodes = _build(calls)
    f = _to_callable(g, _activation(nodes))

    result = f({}, {x_source: 5}, frozenset({"A"}))  # caller provides color A

    assert calls == ["A"]  # only A ran; B's boundary default flowed into make_and
    assert result[nodes["combine"]] == (("A", 5), ("B", "EMPTY"))


def test_other_initial_color():
    calls = []
    g, x_source, nodes = _build(calls)
    f = _to_callable(g, _activation(nodes))

    result = f({}, {x_source: 7}, frozenset({"B"}))

    assert calls == ["B"]
    assert result[nodes["combine"]] == (("A", "EMPTY"), ("B", 7))


def test_no_initial_color_skips_all_colored():
    calls = []
    g, x_source, nodes = _build(calls)
    f = _to_callable(g, _activation(nodes))

    result = f({}, {x_source: 5})  # no color -> only no-color nodes run

    assert calls == []
    assert result[nodes["combine"]] == (("A", "EMPTY"), ("B", "EMPTY"))



def test_interior_node_prunes_when_skipped():
    """A colored node with NO boundary default prunes (absent) when inactive."""
    calls = []

    def worker(x):
        calls.append("worker")
        return ("worked", x)

    x_source = graph.make_source()
    g = composers.compose_left_future(x_source, worker, None, None)
    worker_node = graph.make_computation_node(worker)

    activation = run.NodeActivation(
        node_to_colors={worker_node: frozenset({"A"})}, boundary_defaults={}
    )
    f = _to_callable(g, activation)

    skipped = f({}, {x_source: 9}, frozenset({"B"}))  # B active -> worker(A) prunes
    assert calls == []
    assert worker_node not in skipped

    ran = f({}, {x_source: 9}, frozenset({"A"}))  # A active -> worker runs
    assert calls == ["worker"]
    assert ran[worker_node] == ("worked", 9)


def test_sync_change_event_restarts():
    """A node emits ChangeActiveColors; with no initial color the first pass skips
    the colored node, the event restarts the run, and it then runs."""
    calls = []

    def router(sel):
        return run.ChangeActiveColors(frozenset({sel}))  # the change-color event

    def worker(x):
        calls.append("worker")
        return ("worked", x)

    sel_source = graph.make_source()
    x_source = graph.make_source()
    g = base_types.merge_graphs(
        composers.compose_left_future(sel_source, router, None, None),
        composers.compose_left_future(x_source, worker, None, None),
    )
    worker_node = graph.make_computation_node(worker)
    activation = run.NodeActivation(
        node_to_colors={worker_node: frozenset({"A"})}, boundary_defaults={}
    )
    f = _to_callable(g, activation)

    # No initial color -> worker(A) skipped on pass 0; router emits A -> restart.
    result = f({}, {sel_source: "A", x_source: 9})
    assert calls == ["worker"]
    assert result[worker_node] == ("worked", 9)


# --------------------------------------------------------------------------- #
# Async: the change-color event is downstream of an async VIC, so it isn't known
# when the colored nodes would first run -> the run starts over once the event
# is observed.
# --------------------------------------------------------------------------- #
def _build_async():
    calls = []

    async def vic(utterance):
        await asyncio.sleep(0)  # genuinely async
        return utterance  # a color index, or None for "no skill"

    def chosen_skill_id(v):
        return v  # raw, consumed by select

    def route_color(v):  # the change-color event
        return run.ChangeActiveColors(frozenset() if v is None else frozenset({v}))

    def skill_a(x):
        calls.append("A")
        return ("A", x)

    def skill_b(x):
        calls.append("B")
        return ("B", x)

    def aggregate(a, b):
        return [a, b]

    def select(chosen, agg):
        return ("NONE",) if chosen is None else agg[chosen]

    u = graph.make_source()
    s = graph.make_source()
    g = base_types.merge_graphs(
        composers.compose_left_future(u, vic, None, None),
        composers.compose_left_unary(vic, chosen_skill_id),
        composers.compose_left_unary(vic, route_color),
        composers.compose_left_future(s, skill_a, None, None),
        composers.compose_left_future(s, skill_b, None, None),
        composers.compose_dict(aggregate, {"a": skill_a, "b": skill_b}),
        composers.compose_dict(select, {"chosen": chosen_skill_id, "agg": aggregate}),
    )
    nodes = {
        "a": graph.make_computation_node(skill_a),
        "b": graph.make_computation_node(skill_b),
        "select": graph.make_computation_node(select),
    }
    return g, u, s, nodes, calls


def _async_activation(nodes):
    return run.NodeActivation(
        node_to_colors={nodes["a"]: frozenset({0}), nodes["b"]: frozenset({1})},
        boundary_defaults={nodes["a"]: ("A", "EMPTY"), nodes["b"]: ("B", "EMPTY")},
    )


async def test_async_change_event_routes_to_chosen_skill():
    g, u, s, nodes, calls = _build_async()
    f = _to_callable(g, _async_activation(nodes))

    result = await f({}, {u: 0, s: 5})  # no initial color; event resolves skill 0
    assert calls == ["A"]
    assert result[nodes["select"]] == ("A", 5)


async def test_async_greeting_no_skill_skips_all():
    g, u, s, nodes, calls = _build_async()
    f = _to_callable(g, _async_activation(nodes))

    result = await f({}, {u: None, s: 5})  # event resolves empty -> skip all colored
    assert calls == []
    assert result[nodes["select"]] == ("NONE",)


async def test_async_initial_color_single_pass():
    g, u, s, nodes, calls = _build_async()
    f = _to_callable(g, _async_activation(nodes))

    # Caller seeds color 0 and the route event agrees -> single pass, no restart.
    result = await f({}, {u: 0, s: 5}, frozenset({0}))
    assert calls == ["A"]
    assert result[nodes["select"]] == ("A", 5)


async def test_async_reroute_restarts():
    g, u, s, nodes, calls = _build_async()
    f = _to_callable(g, _async_activation(nodes))

    # Caller seeds color 0, but the route event resolves color 1 -> restart to B.
    result = await f({}, {u: 1, s: 5}, frozenset({0}))
    assert calls == ["A", "B"]  # A ran on the seeded pass, then restarted to B
    assert result[nodes["select"]] == ("B", 5)



# --------------------------------------------------------------------------- #
# Composing declarations: independent declarers (e.g. one per routing context) each
# contribute the colors they resolved; the runner activates the UNION observed in
# the pass. Two declarers resolving different skills in one turn activate BOTH --
# the multi-routing-context bug shape (one declaration silently cancelling the
# other) cannot happen.
# --------------------------------------------------------------------------- #
def _build_two_declarers(calls, colors_one, colors_two):
    def declarer_one(v):
        return run.ChangeActiveColors(colors_one)

    def declarer_two(v):
        return run.ChangeActiveColors(colors_two)

    def skill_a(x):
        calls.append("A")
        return ("A", x)

    def skill_b(x):
        calls.append("B")
        return ("B", x)

    u = graph.make_source()
    s = graph.make_source()
    g = base_types.merge_graphs(
        composers.compose_left_future(u, declarer_one, None, None),
        composers.compose_left_future(u, declarer_two, None, None),
        composers.compose_left_future(s, skill_a, None, None),
        composers.compose_left_future(s, skill_b, None, None),
    )
    activation = run.NodeActivation(
        node_to_colors={
            graph.make_computation_node(skill_a): frozenset({"A"}),
            graph.make_computation_node(skill_b): frozenset({"B"}),
        },
        boundary_defaults={},
    )
    return g, u, s, activation


def test_sync_declarations_union():
    calls = []
    g, u, s, activation = _build_two_declarers(
        calls, frozenset({"A"}), frozenset({"B"})
    )
    f = _to_callable(g, activation)

    f({}, {u: 1, s: 5})
    assert sorted(calls) == ["A", "B"]  # both declared colors activated


def test_sync_empty_declaration_does_not_veto():
    # A declarer that resolved nothing contributes nothing; the other's color runs.
    calls = []
    g, u, s, activation = _build_two_declarers(
        calls, frozenset({"A"}), frozenset()
    )
    f = _to_callable(g, activation)

    f({}, {u: 1, s: 5})
    assert calls == ["A"]


async def test_async_declarations_union():
    calls = []

    async def vic(v):
        await asyncio.sleep(0)
        return v

    def declarer_one(v):
        return run.ChangeActiveColors(frozenset({"A"}))

    def declarer_two(v):
        return run.ChangeActiveColors(frozenset({"B"}))

    def skill_a(x):
        calls.append("A")
        return ("A", x)

    def skill_b(x):
        calls.append("B")
        return ("B", x)

    u = graph.make_source()
    s = graph.make_source()
    g = base_types.merge_graphs(
        composers.compose_left_future(u, vic, None, None),
        composers.compose_left_unary(vic, declarer_one),
        composers.compose_left_unary(vic, declarer_two),
        composers.compose_left_future(s, skill_a, None, None),
        composers.compose_left_future(s, skill_b, None, None),
    )
    activation = run.NodeActivation(
        node_to_colors={
            graph.make_computation_node(skill_a): frozenset({"A"}),
            graph.make_computation_node(skill_b): frozenset({"B"}),
        },
        boundary_defaults={},
    )
    f = _to_callable(g, activation)

    await f({}, {u: 1, s: 5})
    assert sorted(calls) == ["A", "B"]  # both declared colors activated
