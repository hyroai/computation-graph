import asyncio
from typing import Dict

import gamla
import pytest

from computation_graph import base_types, composers, graph, graph_runners, legacy, run
from computation_graph.composers import duplication, memory


def _infer_graph_sink_excluding_terminals(
    edges: base_types.GraphType,
) -> base_types.ComputationNode:
    leaves = gamla.pipe(
        edges, graph.get_leaves, gamla.remove(base_types.node_is_terminal), tuple
    )
    assert len(leaves) == 1, f"computation graph has more than one sink: {leaves}"
    return gamla.head(leaves)


def _node1(arg1):
    return f"node1({arg1})"


async def _node1_async(arg1):
    await asyncio.sleep(0.1)
    return f"node1({arg1})"


def _node2(arg1):
    return f"node2({arg1})"


def _node3(arg1, arg2):
    return f"node3(arg1={arg1}, arg2={arg2})"


def _node_that_raises():
    raise base_types.SkipComputationError


def _merger(args, side_effects):
    return "[" + ",".join(args) + f"], side_effects={side_effects}"


def _next_int(x):
    if x is None:
        return 0
    return x + 1


def _reducer_node(arg1, cur_int):
    return arg1 + f" cur_int={cur_int + 1}"


def test_simple():
    for v in ["root", None]:
        assert (
            graph_runners.unary(
                composers.compose_left_unary(_node1, _node2), _node1, _node2
            )(v)
            == f"node2(node1({v}))"
        )


async def test_async_run_as_soon_as_possible(capsys):
    # This test is to make sure that the async node runs as soon as possible.
    # nodes in different topological layers should run concurrently if not dependent on each other.
    async def concurrent1(x):
        print("start concurrent1")  # noqa
        await asyncio.sleep(0.1)
        print("end concurrent1")  # noqa
        return "concurrent1"

    async def concurrent2(x):
        print("start concurrent2")  # noqa
        await asyncio.sleep(0.1)
        print("end concurrent2")  # noqa
        return "concurrent2"

    def sink(x, y):
        return f"x={x}, y={y}"

    g = base_types.merge_graphs(
        composers.compose_unary(concurrent1, lambda: "x"),
        composers.compose_unary(concurrent2, lambda y: y, lambda: "y"),
        composers.compose_left(concurrent1, sink, key="x"),
        composers.compose_left(concurrent2, sink, key="y"),
    )

    await graph_runners.nullary(g, sink)

    # Capture the output
    captured = capsys.readouterr()

    # Assertions
    assert "start concurrent1" in captured.out
    assert "start concurrent2" in captured.out
    assert "end concurrent1" in captured.out
    assert "end concurrent2" in captured.out
    assert captured.out.index("start concurrent1") < captured.out.index(
        "end concurrent1"
    )
    assert captured.out.index("start concurrent2") < captured.out.index(
        "end concurrent2"
    )
    assert captured.out.index("start concurrent1") < captured.out.index(
        "end concurrent2"
    )
    assert captured.out.index("start concurrent2") < captured.out.index(
        "end concurrent1"
    )


async def test_simple_async():
    assert (
        await graph_runners.unary(
            composers.compose_left_unary(_node1_async, _node2), _node1_async, _node2
        )("hi")
        == "node2(node1(hi))"
    )


def test_kwargs():
    def source(x):
        return x

    assert (
        graph_runners.unary(
            base_types.merge_graphs(
                composers.compose_unary(_node1, source),
                composers.compose_unary(_node2, source),
                composers.compose_dict(_node3, {"arg1": _node1, "arg2": _node2}),
            ),
            source,
            _node3,
        )("hi")
        == "node3(arg1=node1(hi), arg2=node2(hi))"
    )


@legacy.handle_state("state", None)
def _node_with_state_as_arg(arg1, state):
    if state is None:
        state = 0
    return legacy.ComputationResult(
        result=arg1 + f" state={state + 1}", state=state + 1
    )


def test_state():
    f = graph_runners.unary_with_state(
        base_types.merge_graphs(
            composers.compose_left(_node1, _node_with_state_as_arg, key="arg1"),
            composers.compose_left_unary(_node_with_state_as_arg, _node2),
        ),
        _node1,
        _node2,
    )
    assert f("x", "y", "z") == "node2(node1(z) state=3)"


def test_self_future_edge():
    f = graph_runners.unary_with_state(
        base_types.merge_graphs(
            composers.compose_dict(
                _reducer_node, {"arg1": _node1, "cur_int": _next_int}
            ),
            composers.compose_unary(_node2, _reducer_node),
            composers.compose_left_future(_next_int, _next_int, "x", None),
        ),
        _node1,
        _node2,
    )
    assert f("x", "y", "z") == "node2(node1(z) cur_int=3)"


def test_empty_result():
    def raises(x):
        del x
        raise base_types.SkipComputationError

    with pytest.raises(KeyError):
        graph_runners.nullary(composers.compose_unary(raises, lambda: "hi"), raises)


def test_optional():
    def raises():
        raise base_types.SkipComputationError

    def sink(x):
        return x

    assert (
        graph_runners.nullary(
            composers.compose_unary(sink, composers.make_optional(raises, None)), sink
        )
        is None
    )


def test_optional_with_future_edge():
    def output(x):
        return x

    def input(x):
        return x

    f = graph_runners.unary_with_state(
        base_types.merge_graphs(
            composers.make_compose(_reducer_node, input, key="arg1"),
            composers.compose_unary(
                output, composers.make_optional(_reducer_node, None)
            ),
            composers.make_compose(_reducer_node, _next_int, key="cur_int"),
            composers.compose_left_future(_next_int, _next_int, None, None),
        ),
        input,
        output,
    )
    assert f("x", "y", "z") == "z cur_int=3"


def test_first():
    def raises():
        raise base_types.SkipComputationError

    assert (
        graph_runners.nullary_infer_sink(
            composers.make_first(raises, lambda: 1, lambda: 2)
        )
        == 1
    )


async def test_raise_handled_from_async():
    async def raises():
        raise base_types.SkipComputationError

    assert (
        await graph_runners.nullary_infer_sink(
            composers.make_first(raises, lambda: 1, lambda: 2)
        )
        == 1
    )


async def test_no_result_for_node_that_raised_handled_exception():
    async def raises(x):
        raise base_types.SkipComputationError

    def sink(x):
        return x

    @graph.make_computation_node
    def source():
        return 4

    res = await run.to_callable_strict(composers.compose_left(source, raises, sink))(
        {}, {}
    )

    assert res[source] == 4
    assert raises not in res
    assert sink not in res


def test_raise_unhandled_exception():
    class MyExceptionError(Exception):
        ...

    def raises():
        raise MyExceptionError("BAD")

    with pytest.raises(MyExceptionError, match="BAD"):
        graph_runners.nullary_infer_sink(composers.make_first(raises, lambda: 1))


async def test_raise_unhandled_exception_async():
    class MyExceptionError(Exception):
        ...

    @composers.compose_left_dict(
        {"x": composers.compose_left_unary(lambda: 1, lambda x: 1)}
    )
    def raises(x):
        raise MyExceptionError("BAD")

    @composers.compose_left_dict({"x": lambda: 1})
    async def not_awaited(x):
        raise base_types.SkipComputationError("DAB")

    with pytest.raises(MyExceptionError, match="BAD"):
        await graph_runners.nullary_infer_sink(
            composers.compose_left_dict({"y": raises, "x": not_awaited}, lambda x, y: 1)
        )
    assert len(asyncio.all_tasks()) == 1


async def test_raise_exception_in_sync_after_async():
    def raises(x):
        raise TypeError("BAD")

    async def async_source():
        return 4

    with pytest.raises(TypeError, match="BAD"):
        await run.to_callable_strict(
            composers.compose_left(async_source, raises, lambda x: 1)
        )({}, {})


async def test_raise_exception_in_async_after_sync():
    async def raises(x):
        raise TypeError("BAD")

    def source():
        return 4

    with pytest.raises(TypeError, match="BAD"):
        await run.to_callable_strict(
            composers.compose_left(source, raises, lambda x: 1)
        )({}, {})


def test_first_all_unactionable():
    def raises():
        raise base_types.SkipComputationError

    with pytest.raises(KeyError):
        graph_runners.nullary_infer_sink(composers.make_first(raises))


def test_first_with_future_edge():
    def input_node(x):
        return x

    f = graph_runners.unary_with_state(
        base_types.merge_graphs(
            composers.make_compose(_reducer_node, input_node, key="arg1"),
            composers.make_compose(_node1, input_node, key="arg1"),
            composers.make_first(_node_that_raises, _reducer_node, _node1),
            composers.make_compose(_reducer_node, _next_int, key="cur_int"),
            composers.make_compose_future(_next_int, _next_int, "x", None),
        ),
        input_node,
        _reducer_node,
    )
    assert f("x", "y", "z") == "z cur_int=3"


def test_and_with_future():
    source1 = graph.make_source()
    source2 = graph.make_source()
    g = base_types.merge_graphs(
        composers.make_and((_reducer_node, _node2, _node1), _merger),
        composers.compose_source(_merger, key="side_effects", source=source2),
        composers.compose_source(_node1, key="arg1", source=source1),
        composers.compose_source(_node2, key="arg1", source=source1),
        composers.compose_source(_reducer_node, key="arg1", source=source1),
        composers.make_compose(_reducer_node, _next_int, key="cur_int"),
        composers.compose_unary_future(_next_int, _next_int, None),
    )
    assert (
        graph_runners.variadic_stateful_infer_sink(g)(
            {source1: "root", source2: "bla"},
            {source1: "root", source2: "bla"},
            {source1: "root", source2: "bla"},
        )
        == "[root cur_int=3,node2(root),node1(root)], side_effects=bla"
    )


def test_and_with_unactionable():
    source1 = graph.make_source()
    source2 = graph.make_source()
    g = base_types.merge_graphs(
        composers.make_and((_reducer_node, _node_that_raises), _merger),
        composers.compose_source(_merger, key="side_effects", source=source2),
        composers.compose_source(_reducer_node, key="arg1", source=source1),
        composers.make_compose(_reducer_node, _next_int, key="cur_int"),
        composers.compose_unary_future(_next_int, _next_int, None),
    )
    with pytest.raises(KeyError):
        graph_runners.variadic_infer_sink(g)({source1: "root", source2: "bla"})


def test_or():
    def merger(args):
        return " ".join(map(str, args))

    f = graph_runners.variadic_stateful_infer_sink(
        base_types.merge_graphs(
            composers.make_or((_next_int, lambda: "node1", _node_that_raises), merger),
            composers.compose_unary_future(_next_int, _next_int, 0),
        )
    )

    assert f({}, {}, {}) == "3 node1"


def test_compose():
    assert (
        graph_runners.nullary_infer_sink(composers.make_compose(_node1, lambda: "hi"))
        == "node1(hi)"
    )


def test_compose_with_future_edge():
    f = graph_runners.unary_with_state(
        base_types.merge_graphs(
            composers.make_compose(_node1, _node2),
            composers.make_compose(_reducer_node, _node1, key="arg1"),
            composers.make_compose(_reducer_node, _next_int, key="cur_int"),
            composers.make_compose_future(_next_int, _next_int, None, None),
        ),
        _node2,
        _reducer_node,
    )
    assert f("hi", "hi", "hi") == "node1(node2(hi)) cur_int=3"


def test_optional_memory_sometimes_raises():
    def sometimes_raises(x, cur_int):
        if x == "fail":
            raise base_types.SkipComputationError
        return x + f" state={cur_int + 1}"

    def input_source(x):
        return x

    f = graph_runners.unary_with_state_infer_sink(
        base_types.merge_graphs(
            composers.make_compose(sometimes_raises, input_source, key="x"),
            composers.make_optional(sometimes_raises, None),
            composers.make_compose(sometimes_raises, _next_int, key="cur_int"),
            composers.compose_unary_future(_next_int, _next_int, None),
        ),
        input_source,
    )
    assert f("hi", "fail", "hi") == "hi state=3"


def test_first_first():
    def node1():
        return "node1"

    assert (
        graph_runners.nullary_infer_sink(
            composers.make_first(
                _node_that_raises,
                composers.make_first(_node_that_raises, node1, lambda: "node2"),
                node1,
            )
        )
        == "node1"
    )


def test_compose_with_node_already_in_graph():
    def node1():
        return "node1"

    def sink(x, y):
        return f"x={x} y={y}"

    def merger(args):
        return " ".join(args)

    assert (
        graph_runners.nullary_infer_sink(
            composers.make_compose(
                composers.make_compose(sink, node1, key="x"),
                composers.make_and((lambda: "node2", node1), merge_fn=merger),
                key="y",
            )
        )
        == "x=node1 y=node2 node1"
    )


def test_first_with_subgraph_that_raises():
    def raises():
        raise base_types.SkipComputationError

    def node2(x):
        return x

    def node1():
        return "node1"

    assert (
        graph_runners.nullary_infer_sink(
            composers.make_first(composers.compose_unary(node2, raises), node1)
        )
        == "node1"
    )


def test_or_with_sink_that_raises():
    def raises():
        raise base_types.SkipComputationError

    def merge(args):
        if not args:
            raise base_types.SkipComputationError
        return ",".join(args)

    assert (
        graph_runners.nullary_infer_sink(
            composers.make_or((raises, lambda: "node1"), merge_fn=merge)
        )
        == "node1"
    )


def test_unambiguous_composition_using_terminal():
    terminal = graph.make_terminal("1", lambda x: x[0])

    def source():
        return 1

    with pytest.raises(AssertionError):
        composers.compose_unary(
            lambda x: x + 1,
            base_types.merge_graphs(
                composers.compose_unary(lambda x: x + 1, source),
                composers.compose_unary(lambda x: x, source),
            ),
        )

    g = composers.compose_unary(
        lambda x: x + 1,
        base_types.merge_graphs(
            composers.compose_unary(lambda x: x + 1, source),
            composers.compose_unary(terminal, source),
        ),
    )
    x = run.to_callable_strict(g)({}, {})
    assert x[terminal] == 1
    assert x[_infer_graph_sink_excluding_terminals(g)] == 3


def test_two_terminals():
    terminal1 = graph.make_terminal("1", lambda x: x)
    terminal2 = graph.make_terminal("2", lambda x: x)
    result = graph_runners.unary_bare(
        base_types.merge_graphs(
            composers.compose_unary(terminal1, composers.make_compose(_node2, _node1)),
            composers.compose_unary(terminal2, _node1),
        ),
        _node1,
    )("hi")
    assert result[terminal1][0] == "node2(node1(hi))"
    assert result[terminal2][0] == "node1(hi)"


def test_two_paths_succeed():
    source = graph.make_source()
    terminal1 = graph.make_terminal("1", lambda x: x)
    terminal2 = graph.make_terminal("2", lambda x: x)
    result = graph_runners.variadic_bare(
        base_types.merge_graphs(
            composers.make_first(
                composers.compose_source_unary(_node1, source),
                composers.compose_unary(
                    terminal1, composers.compose_source_unary(_node2, source)
                ),
            ),
            composers.compose_unary(terminal2, _node1),
        )
    )({source: "hi"})
    assert result[terminal1][0] == "node2(hi)"
    assert result[terminal2][0] == "node1(hi)"


def test_double_star_signature_considered_unary():
    sink = gamla.juxt(
        lambda some_argname: some_argname + 1,
        lambda different_argname: different_argname * 2,
    )
    assert graph_runners.nullary(composers.make_compose(sink, lambda: 3), sink) == (
        4,
        6,
    )


def test_type_safety_messages(caplog, monkeypatch):
    monkeypatch.setenv(base_types.COMPUTATION_GRAPH_DEBUG_ENV_KEY, "j")

    def f(x) -> int:  # Bad typing!
        return "hello " + x

    assert (
        graph_runners.nullary(composers.make_compose(f, lambda: "world"), f)
        == "hello world"
    )
    assert "TypeError" in caplog.text


def test_type_safety_messages_no_overtrigger(caplog, monkeypatch):
    monkeypatch.setenv(base_types.COMPUTATION_GRAPH_DEBUG_ENV_KEY, "h")

    def f(x) -> str:
        return "hello " + x

    assert (
        graph_runners.nullary(composers.make_compose(f, lambda: "world"), f)
        == "hello world"
    )
    assert "TypeError" not in caplog.text


def test_anonymous_composition_type_safety():
    def f() -> str:
        pass

    def g(x: int):
        pass

    with pytest.raises(base_types.ComputationGraphTypeError):
        composers.make_compose(g, f)


def test_named_composition_type_safety():
    def f() -> str:
        pass

    def g(x: int):
        pass

    with pytest.raises(base_types.ComputationGraphTypeError):
        composers.make_compose(g, f, key="x")


def _multiply(a, b):
    if b:
        return a * b
    return a


def _plus_1(y):
    return y + 1


async def _plus_1_async(y):
    await asyncio.sleep(0.1)
    return y + 1


def _times_2(x):
    return x * 2


def _sum(args):
    return sum(args)


def test_future_edges():
    graph_runners.unary_with_state_and_expectations(
        base_types.merge_graphs(
            composers.compose_unary(_plus_1, _times_2),
            composers.make_compose(_multiply, _plus_1, key="a"),
            composers.make_compose_future(_multiply, _times_2, "b", None),
        ),
        _times_2,
        _multiply,
    )([[3, 7], [3, 42]])


def test_future_edges_with_circuit():
    def some_input(x):
        return x

    f = graph_runners.unary_with_state(
        base_types.merge_graphs(
            composers.make_compose(_plus_1, _multiply),
            composers.make_compose(_times_2, _plus_1),
            composers.make_compose(_multiply, some_input, key="a"),
            composers.make_compose_future(_multiply, _times_2, "b", None),
        ),
        some_input,
        _times_2,
    )
    assert f(3, 3) == 50


def test_sink_with_incoming_future_edge():
    def f(x):
        return x

    def g(x, y):
        if y is None:
            y = 4
        return f"x={x}, y={y}"

    f = graph_runners.unary(
        base_types.merge_graphs(
            composers.make_compose(g, f, key="x"),
            composers.make_compose_future(g, g, "y", None),
        ),
        f,
        g,
    )
    assert f(3) == "x=3, y=4"


def test_compose_future():
    a = graph.make_source()
    b = graph.make_source()
    c = graph.make_source()
    graph_runners.variadic_with_state_and_expectations(
        base_types.merge_graphs(
            composers.compose_source_unary(_plus_1, c),
            composers.compose_source_unary(_times_2, b),
            composers.compose_source(_multiply, "a", a),
            composers.make_compose_future(
                _multiply,
                composers.make_and([_plus_1, _times_2, _multiply], merge_fn=_sum),
                "b",
                None,
            ),
        ),
        _sum,
    )(([[{a: 2, b: 2, c: 2}, 9], [{a: 2, b: 2, c: 2}, 25]]))


async def test_compose_future_async():
    a = graph.make_source()
    b = graph.make_source()
    c = graph.make_source()
    await graph_runners.variadic_with_state_and_expectations(
        base_types.merge_graphs(
            composers.compose_source_unary(_plus_1_async, c),
            composers.compose_source_unary(_times_2, b),
            composers.compose_source(_multiply, "a", a),
            composers.make_compose_future(
                _multiply,
                composers.make_and([_plus_1_async, _times_2, _multiply], merge_fn=_sum),
                "b",
                None,
            ),
        ),
        _sum,
    )(([[{a: 2, b: 2, c: 2}, 9], [{a: 2, b: 2, c: 2}, 25]]))


def test_dont_duplicate_sources():
    a = graph.make_source()
    assert (
        graph_runners.variadic_infer_sink(
            duplication.duplicate_function_or_graph(
                composers.compose_source_unary(_plus_1, a)
            )
        )({a: 2})
        == 3
    )


def test_badly_composed_graph_raises():
    with pytest.raises(AssertionError):
        run.to_callable_strict(
            composers.make_compose(lambda x, y: x + y, lambda: 1, key="x")
        )


def test_memory_persists_when_unactionable():
    def input_node(x):
        return x

    def output_node(x):
        return x

    def skipper(upstream, x):
        return x or upstream

    remember_first = memory.with_state("x", None, skipper)
    skip_or_passthrough = (
        lambda input: input
        if input != "skip state"
        else gamla.just_raise(base_types.SkipComputationError)
    )
    graph_runners.unary_with_state_and_expectations(
        composers.compose_left(
            composers.make_first(
                composers.make_compose(
                    composers.compose_left(
                        skip_or_passthrough, remember_first, key="upstream"
                    ),
                    input_node,
                    key="input",
                ),
                lambda: "state skipped",
            ),
            output_node,
        ),
        input_node,
        output_node,
    )(
        [
            ["remember this", "remember this"],
            ["skip state", "state skipped"],
            ["recall", "remember this"],
        ]
    )


def test_replace_source():
    a = graph.make_source()

    assert graph.replace_source(_node1, _node1_async)(
        base_types.merge_graphs(
            composers.compose_source_unary(_node1, a),
            composers.compose_left_unary(_node1, _node2),
        )
    ) == base_types.merge_graphs(
        composers.compose_source_unary(_node1, a),
        composers.compose_left_unary(_node1_async, _node2),
    )


def test_replace_source_with_args():
    assert graph.replace_source(_node1, _node1_async)(
        (
            base_types.ComputationEdge(
                is_future=False,
                priority=0,
                source=None,
                args=(
                    graph.make_computation_node(_node1),
                    graph.make_computation_node(_node2),
                ),
                destination=graph.make_computation_node(_merger),
                key="args",
            ),
        )
    ) == frozenset(
        (
            base_types.ComputationEdge(
                is_future=False,
                priority=0,
                source=None,
                args=(
                    graph.make_computation_node(_node1_async),
                    graph.make_computation_node(_node2),
                ),
                destination=graph.make_computation_node(_merger),
                key="args",
            ),
        )
    )


def test_replace_source_with_graph():
    a = graph.make_source()

    assert frozenset(
        graph.replace_source(
            _node1, composers.compose_left_unary(_node1_async, _next_int)
        )(
            base_types.merge_graphs(
                composers.compose_source_unary(_node1, a),
                composers.compose_left_unary(_node1, _node2),
            )
        )
    ) == base_types.merge_graphs(
        composers.compose_source_unary(_node1, a),
        composers.compose_left_unary(_next_int, _node2),
        composers.compose_left_unary(_node1_async, _next_int),
    )


def test_replace_source_that_doesnt_exist():
    a = graph.make_source()

    assert graph.replace_source(
        lambda x: x, composers.compose_left_unary(_node1_async, _next_int)
    )(
        base_types.merge_graphs(
            composers.compose_source_unary(_node1, a),
            composers.compose_left_unary(_node1, _node2),
        )
    ) == base_types.merge_graphs(
        composers.compose_source_unary(_node1, a),
        composers.compose_left_unary(_node1, _node2),
    )


def test_replace_destination():
    assert graph.replace_destination(_node1, _node1_async)(
        (
            base_types.ComputationEdge(
                is_future=False,
                priority=0,
                source=graph.make_computation_node(_node2),
                args=(),
                destination=graph.make_computation_node(_node1),
                key="arg1",
            ),
        )
    ) == base_types.merge_graphs(
        (
            base_types.ComputationEdge(
                is_future=False,
                priority=0,
                source=graph.make_computation_node(_node2),
                args=(),
                destination=graph.make_computation_node(_node1_async),
                key="arg1",
            ),
        )
    )


def test_replace_node():
    a = graph.make_source()

    assert graph.replace_node(_node1, _node1_async)(
        base_types.merge_graphs(
            composers.compose_source_unary(_node1, a),
            composers.compose_left_unary(_node1, _node2),
        )
    ) == base_types.merge_graphs(
        composers.compose_source_unary(_node1_async, a),
        composers.compose_left_unary(_node1_async, _node2),
    )


def test_ambig_edges_assertion_in_merge_graphs_active_only_when_env_var_is_active(
    monkeypatch,
):
    def a(x):
        pass

    monkeypatch.delenv(base_types.COMPUTATION_GRAPH_DEBUG_ENV_KEY, raising=False)
    base_types.merge_graphs(
        composers.compose_left_unary(lambda: 1, a),
        composers.compose_left_unary(lambda: 1, a),
    )

    monkeypatch.setenv(base_types.COMPUTATION_GRAPH_DEBUG_ENV_KEY, "1")
    with pytest.raises(
        Exception, match=r".*There are multiple edges with the same destination.*"
    ):
        base_types.merge_graphs(
            composers.compose_left_unary(lambda: 1, a),
            composers.compose_left_unary(lambda: 1, a),
        )


def a():
    pass


def b(x):
    pass


def c(x):
    pass


def d(x, y):
    pass


def kuky():
    pass


def kuku():
    pass


t = graph.make_terminal("t", lambda x: x)

g = base_types.merge_graphs(
    composers.compose_left(a, c),
    composers.compose_left(c, d, key="x"),
    composers.compose_left(b, d, key="y"),
    composers.compose_left_future(d, b, "x", "bla"),
    composers.compose_left(a, t),
)


@pytest.mark.parametrize(
    "to_replace,expected_edges_strs",
    [
        pytest.param(
            {a: kuky},
            {
                "kuky----x---->duplicate of c",
                "kuky----x---->t",
                "duplicate of c----x---->duplicate of d",
                "when_memory_unavailable----x---->duplicate of b",
                "duplicate of d....x....>duplicate of b",
                "duplicate of b----y---->duplicate of d",
            },
            id="replace source node",
        ),
        pytest.param(
            {c: kuky},
            {
                "a----x---->kuky",
                "a----x---->t",
                "kuky----x---->duplicate of d",
                "when_memory_unavailable----x---->duplicate of b",
                "duplicate of d....x....>duplicate of b",
                "duplicate of b----y---->duplicate of d",
            },
            id="replace node not in cycle",
        ),
        pytest.param(
            {b: kuky},
            {
                "a----x---->c",
                "a----x---->t",
                "c----x---->duplicate of d",
                "when_memory_unavailable----x---->kuky",
                "duplicate of d....x....>kuky",
                "kuky----y---->duplicate of d",
            },
            id="replace node in cycle",
        ),
        pytest.param(
            {a: kuku, b: kuky},
            {
                "duplicate of c----x---->duplicate of d",
                "duplicate of d....x....>kuky",
                "kuku----x---->duplicate of c",
                "kuku----x---->t",
                "kuky----y---->duplicate of d",
                "when_memory_unavailable----x---->kuky",
            },
            id="replace multiple nodes - duplicate reachables once",
        ),
    ],
)
def test_safe_replace_node(
    to_replace: Dict[base_types.CallableOrNode, base_types.CallableOrNodeOrGraph],
    expected_edges_strs: str,
):
    assert expected_edges_strs == {
        str(e) for e in duplication.safe_replace_sources(to_replace, g)
    }
