"""Tests for sync-chain fusion (COMPUTATION_GRAPH_SYNC_FUSION).

The flag is read at graph compilation time, so each test sets/clears the env
var before calling `run.to_callable_strict`.
"""
import asyncio

import gamla
import pytest

from computation_graph import base_types, composers, graph, run, sync_fusion

_FUSION_KEY = sync_fusion.COMPUTATION_GRAPH_SYNC_FUSION_ENV_KEY


def _fusion_on(monkeypatch):
    monkeypatch.setenv(_FUSION_KEY, "1")


def _fusion_off(monkeypatch):
    monkeypatch.delenv(_FUSION_KEY, raising=False)


def _sync_and_downstream(edges):
    all_nodes = graph.get_all_nodes(edges)
    async_nodes = {
        n for n in all_nodes if asyncio.iscoroutinefunction(n.func)
    }
    downstream = set(
        gamla.graph_traverse_many(async_nodes, graph.traverse_forward(edges))
    )
    return (all_nodes - async_nodes) & downstream


async def _async_root():
    return 1


def test_chain_detection():
    def s1(x):
        return x

    def s2(x):
        return x

    def s3(x):
        return x

    def other(x):
        return x

    def fan_in(x, y):
        return (x, y)

    edges = base_types.merge_graphs(
        composers.compose_left_unary(_async_root, s1),
        composers.compose_left_unary(s1, s2),
        composers.compose_left_unary(s2, s3),
        composers.compose_left_unary(_async_root, other),
        composers.compose_left(s3, fan_in, key="x"),
        composers.compose_left(other, fan_in, key="y"),
    )
    chains = sync_fusion._sync_chains(tuple(edges), _sync_and_downstream(edges))
    node = graph.make_computation_node
    # fan_in has two sync producers so it cannot join a chain; `other` has a
    # single consumer but is alone (its consumer has in-degree 2), so no
    # 2-node chain contains it.
    assert chains == ((node(s1), node(s2), node(s3)),)


async def test_fused_results_equal_unfused(monkeypatch):
    def s1(x):
        return x + 1

    def s2(x):
        return x * 2

    def skips(x):
        raise base_types.SkipComputationError

    def after_skip(x):
        return x

    async def consumer_a(x):
        return ("a", x)

    async def consumer_b(x):
        return ("b", x)

    g = base_types.merge_graphs(
        composers.compose_left_unary(_async_root, s1),
        composers.compose_left_unary(s1, s2),
        composers.compose_left_unary(s2, skips),
        composers.compose_left_unary(skips, after_skip),
        composers.compose_left_unary(s2, consumer_a),
        composers.compose_left_unary(s2, consumer_b),
    )
    _fusion_off(monkeypatch)
    unfused = await run.to_callable_strict(g)({}, {})
    _fusion_on(monkeypatch)
    fused = await run.to_callable_strict(g)({}, {})
    assert fused == unfused
    assert fused[graph.make_computation_node(consumer_a)] == ("a", 4)
    assert graph.make_computation_node(after_skip) not in fused


async def test_chain_nodes_share_one_task_only_when_fused(monkeypatch):
    tasks = []

    def s1(x):
        tasks.append(asyncio.current_task())
        return x

    def s2(x):
        tasks.append(asyncio.current_task())
        return x

    g = base_types.merge_graphs(
        composers.compose_left_unary(_async_root, s1),
        composers.compose_left_unary(s1, s2),
    )

    _fusion_on(monkeypatch)
    await run.to_callable_strict(g)({}, {})
    assert tasks[0] is tasks[1]

    tasks.clear()
    _fusion_off(monkeypatch)
    await run.to_callable_strict(g)({}, {})
    assert tasks[0] is not tasks[1]


async def test_skip_mid_chain_propagates_and_unrelated_chain_survives(monkeypatch):
    _fusion_on(monkeypatch)

    def s1(x):
        return x + 1

    def s2(x):
        raise base_types.SkipComputationError

    def s3(x):
        return x * 10

    def t1(x):
        return x + 100

    def t2(x):
        return x + 1000

    g = base_types.merge_graphs(
        composers.compose_left_unary(_async_root, s1),
        composers.compose_left_unary(s1, s2),
        composers.compose_left_unary(s2, s3),
        composers.compose_left_unary(_async_root, t1),
        composers.compose_left_unary(t1, t2),
    )
    results = await run.to_callable_strict(g)({}, {})
    node = graph.make_computation_node
    assert results[node(s1)] == 2
    assert node(s2) not in results
    assert node(s3) not in results
    assert results[node(t2)] == 1101


async def test_failed_external_async_input_skips_chain(monkeypatch):
    _fusion_on(monkeypatch)

    async def async_skips():
        raise base_types.SkipComputationError

    def c1(x):
        return x + 1

    def c2(x):
        return x + 2

    def d1(x):
        return x + 3

    def d2(x):
        return x + 4

    g = base_types.merge_graphs(
        composers.compose_left_unary(async_skips, c1),
        composers.compose_left_unary(c1, c2),
        composers.compose_left_unary(_async_root, d1),
        composers.compose_left_unary(d1, d2),
    )
    results = await run.to_callable_strict(g)({}, {})
    node = graph.make_computation_node
    assert node(c1) not in results
    assert node(c2) not in results
    assert results[node(d2)] == 8


async def test_unhandled_exception_mid_chain_raises(monkeypatch):
    _fusion_on(monkeypatch)

    def s1(x):
        return x

    def boom(x):
        raise TypeError("BAD")

    def s3(x):
        return x

    def t1(x):
        return x

    def t2(x):
        return x

    g = base_types.merge_graphs(
        composers.compose_left_unary(_async_root, s1),
        composers.compose_left_unary(s1, boom),
        composers.compose_left_unary(boom, s3),
        composers.compose_left_unary(_async_root, t1),
        composers.compose_left_unary(t1, t2),
    )
    with pytest.raises(TypeError, match="BAD"):
        await asyncio.wait_for(run.to_callable_strict(g)({}, {}), timeout=5)
    assert len(asyncio.all_tasks()) == 1


async def test_fanout_from_chain_output(monkeypatch):
    _fusion_on(monkeypatch)

    def s1(x):
        return x + 1

    def s2(x):
        return x * 2

    async def consumer_a(x):
        return ("a", x)

    async def consumer_b(x):
        return ("b", x)

    g = base_types.merge_graphs(
        composers.compose_left_unary(_async_root, s1),
        composers.compose_left_unary(s1, s2),
        composers.compose_left_unary(s1, consumer_a),
        composers.compose_left_unary(s1, consumer_b),
    )
    results = await run.to_callable_strict(g)({}, {})
    node = graph.make_computation_node
    assert results[node(s2)] == 4
    assert results[node(consumer_a)] == ("a", 2)
    assert results[node(consumer_b)] == ("b", 2)


async def test_reconvergence_through_async_does_not_deadlock(monkeypatch):
    _fusion_on(monkeypatch)

    def b(x):
        return x + 1

    async def c(x):
        await asyncio.sleep(0)
        return x * 10

    def d(x, y):
        return (x, y)

    # b and d form a chain in the sync subgraph, but d also depends on b
    # through the async node c. The chain must publish b's result before
    # awaiting c, otherwise this deadlocks.
    g = base_types.merge_graphs(
        composers.compose_left_unary(_async_root, b),
        composers.compose_left_unary(b, c),
        composers.compose_left(b, d, key="x"),
        composers.compose_left(c, d, key="y"),
    )
    results = await asyncio.wait_for(run.to_callable_strict(g)({}, {}), timeout=5)
    assert results[graph.make_computation_node(d)] == (2, 20)


async def test_priority_fallback_when_chain_node_skips(monkeypatch):
    _fusion_on(monkeypatch)

    def maybe(x):
        raise base_types.SkipComputationError

    def fallback():
        return "fallback"

    g = composers.compose_left_unary(
        composers.make_first(
            composers.compose_left_unary(_async_root, maybe), fallback
        ),
        lambda x: x,
    )
    results = await run.to_callable_strict(g)({}, {})
    sink = gamla.head(
        n for n in graph.get_leaves(tuple(g)) if not n.is_terminal
    )
    assert results[sink] == "fallback"


async def test_async_fallback_option_inside_chain(monkeypatch):
    _fusion_on(monkeypatch)

    def skips(v):
        raise base_types.SkipComputationError

    async def async_fallback():
        return "af"

    def tail(v):
        return v + "!"

    g = composers.compose_left_unary(
        composers.make_first(
            composers.compose_left_unary(_async_root, skips), async_fallback
        ),
        tail,
    )
    results = await run.to_callable_strict(g)({}, {})
    assert results[graph.make_computation_node(tail)] == "af!"


async def test_state_future_edge_through_chain(monkeypatch):
    _fusion_on(monkeypatch)

    def accumulate(x, prev):
        return (prev or 0) + x

    def fmt(acc):
        return f"acc={acc}"

    g = base_types.merge_graphs(
        composers.compose_left(_async_root, accumulate, key="x"),
        composers.compose_left_future(accumulate, accumulate, "prev", None),
        composers.compose_left_unary(accumulate, fmt),
    )
    f = run.to_callable_strict(g)
    prev = {}
    for expected in ("acc=1", "acc=2", "acc=3"):
        prev = await f(prev, {})
        assert prev[graph.make_computation_node(fmt)] == expected


async def test_debug_equivalence_check(monkeypatch):
    _fusion_on(monkeypatch)
    monkeypatch.setenv(base_types.COMPUTATION_GRAPH_DEBUG_ENV_KEY, "1")

    def s1(x):
        return x + 1

    def skips(x):
        raise base_types.SkipComputationError

    async def consumer(x):
        return ("c", x)

    g = base_types.merge_graphs(
        composers.compose_left_unary(_async_root, s1),
        composers.compose_left_unary(s1, skips),
        composers.compose_left_unary(s1, consumer),
    )
    results = await run.to_callable_strict(g)({}, {})
    assert results[graph.make_computation_node(consumer)] == ("c", 2)
