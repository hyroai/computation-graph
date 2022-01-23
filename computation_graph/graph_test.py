import asyncio
import json

import gamla
import pytest

from computation_graph import base_types, composers, graph, graph_runners, legacy, run

pytestmark = pytest.mark.asyncio


_ROOT_VALUE = "root"


def _node1(arg1):
    return f"node1({arg1})"


async def _node1_async(arg1):
    await asyncio.sleep(0.1)
    return f"node1({arg1})"


def _node2(arg1):
    return f"node2({arg1})"


def _node3(arg1, arg2):
    return f"node3(arg1={arg1}, arg2={arg2})"


def _node4(y, z):
    return f"node4({y}, z={z})"


@gamla.curry
def _curried_node(arg1, arg2):
    return f"curried_node(arg1={arg1}, arg2={arg2})"


def _node_that_raises():
    raise base_types.SkipComputationError


def _merger(args, side_effects):
    return "[" + ",".join(args) + f"], side_effects={side_effects}"


def _merger_that_raises_when_empty(args):
    if not args:
        raise base_types.SkipComputationError
    return "[" + ",".join(args) + "]"


def _node_with_optional_param(optional_param: int = 5):
    return f"node_with_optional_param(optional_param={optional_param})"


def _next_int(x):
    if x is None:
        return 0
    return x + 1


def _reducer_node(arg1, cur_int):
    return arg1 + f" cur_int={cur_int + 1}"


def _sometimes_unactionable_reducer_node(arg1, cur_int):
    if arg1 == "fail":
        raise base_types.SkipComputationError
    return arg1 + f" state={cur_int + 1}"


def test_simple():
    for v in ["root", None]:
        assert (
            graph_runners.unary(
                composers.compose_left_unary(_node1, _node2), _node1, _node2
            )(v)
            == f"node2(node1({v}))"
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
        )(_ROOT_VALUE)
        == f"node3(arg1=node1({_ROOT_VALUE}), arg2=node2({_ROOT_VALUE}))"
    )


@legacy.handle_state("state", None)
def _node_with_state_as_arg(arg1, state):
    if state is None:
        state = 0
    return legacy.LegacyComputationResult(
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
    assert f(_ROOT_VALUE, _ROOT_VALUE, _ROOT_VALUE) == "node2(node1(root) state=3)"


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
    assert f(_ROOT_VALUE, _ROOT_VALUE, _ROOT_VALUE) == "node2(node1(root) cur_int=3)"


def test_empty_result():
    with pytest.raises(KeyError):
        graph_runners.nullary(
            composers.compose_unary(_node_that_raises, lambda: "hi"), _node_that_raises
        )


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
    assert f(_ROOT_VALUE, _ROOT_VALUE, _ROOT_VALUE) == "root cur_int=3"


def test_first():
    def raises():
        raise base_types.SkipComputationError

    assert (
        graph_runners.nullary_infer_sink(
            composers.make_first(raises, lambda: 1, lambda: 2)
        )
        == 1
    )


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
    assert f(_ROOT_VALUE, _ROOT_VALUE, _ROOT_VALUE) == "root cur_int=3"


def test_and_with_future():
    source1 = graph.make_source()
    source2 = graph.make_source()
    g = base_types.merge_graphs(
        composers.make_and((_reducer_node, _node2, _node1), _merger),
        composers.compose_source(_merger, source2, key="side_effects"),
        composers.compose_source(_node1, source1, key="arg1"),
        composers.compose_source(_node2, source1, key="arg1"),
        composers.compose_source(_reducer_node, source1, key="arg1"),
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
        composers.compose_source(_merger, source2, key="side_effects"),
        composers.compose_source(_reducer_node, source1, key="arg1"),
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


def test_graph_wrapping():
    edges = graph.connect_default_terminal(
        composers.make_first(
            _node_that_raises,
            composers.make_and(funcs=(_node1, _node2, _node3), merge_fn=_merger),
            _node1,
        )
    )

    result = run.to_callable(edges, frozenset([base_types.SkipComputationError]))(
        arg1=_ROOT_VALUE, arg2="node3value", side_effects="side_effects"
    )

    assert (
        result.result[graph.DEFAULT_TERMINAL][0]
        == "[node1(root),node2(root),node3(arg1=root, arg2=node3value)], side_effects=side_effects"
    )


def test_node_with_optional_param():
    edges = graph.connect_default_terminal(
        (
            graph.make_standard_edge(
                source=_node_with_optional_param, destination=_node1, key="arg1"
            ),
        )
    )

    result = run.to_callable(edges, frozenset([base_types.SkipComputationError]))()

    assert (
        result.result[graph.DEFAULT_TERMINAL][0]
        == "node1(node_with_optional_param(optional_param=5))"
    )


def test_node_with_bound_optional_param():
    edges = graph.connect_default_terminal(
        (
            graph.make_standard_edge(
                source=_node_with_optional_param, destination=_node1, key="arg1"
            ),
        )
    )

    result = run.to_callable(edges, frozenset([base_types.SkipComputationError]))(
        optional_param=10
    )

    assert (
        result.result[graph.DEFAULT_TERMINAL][0]
        == "node1(node_with_optional_param(optional_param=10))"
    )


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
    assert f(_ROOT_VALUE, _ROOT_VALUE, _ROOT_VALUE) == "node1(node2(root)) cur_int=3"


def test_compose_when_all_arguments_have_a_default():
    edges = graph.connect_default_terminal(
        composers.make_compose(_node_with_optional_param, _node1)
    )

    result = run.to_callable(edges, frozenset([base_types.SkipComputationError]))(
        arg1=_ROOT_VALUE
    )

    assert (
        result.result[graph.DEFAULT_TERMINAL][0]
        == "node_with_optional_param(optional_param=node1(root))"
    )


def test_optional_with_sometimes_unactionable_reducer():
    edges = graph.connect_default_terminal(
        composers.make_optional(
            _sometimes_unactionable_reducer_node, default_value=None
        )
        + (
            graph.make_standard_edge(
                source=_next_int,
                destination=_sometimes_unactionable_reducer_node,
                key="cur_int",
            ),
            graph.make_future_edge(source=_next_int, destination=_next_int),
        )
    )
    cg = run.to_callable(edges, frozenset([base_types.SkipComputationError]))
    result = cg(arg1=_ROOT_VALUE)
    result = cg(arg1="fail", state=result.state)
    result = cg(arg1=_ROOT_VALUE, state=result.state)

    assert result.result[graph.DEFAULT_TERMINAL][0] == "root state=2"


def test_unary_graph_composition():
    inner = composers.make_compose(_node1, _node4)
    edges = graph.connect_default_terminal(composers.make_first(inner))
    result = run.to_callable(edges, frozenset([base_types.SkipComputationError]))(
        y=_ROOT_VALUE, z=10
    )

    assert result.result[graph.DEFAULT_TERMINAL][0] == "node1(node4(root, z=10))"


def test_state_is_serializable():
    edges = graph.connect_default_terminal(
        (
            graph.make_standard_edge(
                source=_node1, destination=_reducer_node, key="arg1"
            ),
            graph.make_standard_edge(
                source=_reducer_node, destination=_node2, key="arg1"
            ),
            graph.make_standard_edge(
                source=_next_int, destination=_reducer_node, key="cur_int"
            ),
            graph.make_future_edge(source=_next_int, destination=_next_int),
        )
    )

    cg = run.to_callable(edges, frozenset([base_types.SkipComputationError]))
    result = cg(arg1=_ROOT_VALUE)
    result = cg(arg1=_ROOT_VALUE, state=result.state)
    result = cg(arg1=_ROOT_VALUE, state=result.state)
    json.dumps(result.state)


def test_compose_compose():

    inner_graph = composers.make_compose(
        _curried_node, _node1, _node3, _node2, key="arg1"
    )
    assert len(inner_graph) == 3

    edges = graph.connect_default_terminal(
        composers.make_compose(inner_graph, _node4, key="arg2")
    )

    result = run.to_callable(edges, frozenset([base_types.SkipComputationError]))(
        y="y", z="z", arg1="arg1"
    )

    assert len(edges) == 6
    assert (
        result.result[graph.DEFAULT_TERMINAL][0]
        == "curried_node(arg1=node1(node3(arg1=node2(arg1), arg2=node4(y, z=z))), arg2=node4(y, z=z))"
    )


def test_compose_after_first():
    edges = graph.connect_default_terminal(
        composers.make_compose(
            composers.make_first(_node_that_raises, _node1, _node2), _node3, key="arg1"
        )
    )
    result = run.to_callable(edges, frozenset([base_types.SkipComputationError]))(
        arg1="arg1", arg2="arg2"
    )
    assert (
        result.result[graph.DEFAULT_TERMINAL][0] == "node1(node3(arg1=arg1, arg2=arg2))"
    )


def test_first_after_compose():
    inner_edges = composers.make_compose(_node1, _node2)

    cg = run.to_callable(
        graph.connect_default_terminal(
            composers.make_first(_node_that_raises, inner_edges, _node1)
        ),
        frozenset([base_types.SkipComputationError]),
    )

    result = cg(arg1="arg1")
    assert result.result[graph.DEFAULT_TERMINAL][0] == "node1(node2(arg1))"


def test_first_first():
    inner_first = composers.make_first(_node_that_raises, _node1, _node2)

    cg = run.to_callable(
        graph.connect_default_terminal(
            composers.make_first(_node_that_raises, inner_first, _node1)
        ),
        frozenset([base_types.SkipComputationError]),
    )

    result = cg(arg1=_ROOT_VALUE)

    assert result.result[graph.DEFAULT_TERMINAL][0] == f"node1({_ROOT_VALUE})"


def test_compose_with_node_already_in_graph():
    inner_edges1 = composers.make_and((_node2, _node1), merge_fn=_merger)
    inner_edges2 = composers.make_compose(_node3, _node1, key="arg1")
    edges = graph.connect_default_terminal(
        composers.make_compose(inner_edges1, inner_edges2, key="arg1")
    )

    result = run.to_callable(edges, frozenset([base_types.SkipComputationError]))(
        arg1=_ROOT_VALUE, arg2="arg2", side_effects="side_effects"
    )

    assert (
        result.result[graph.DEFAULT_TERMINAL][0]
        == "[node2(node3(arg1=node1(root), arg2=arg2)),node1(root)], side_effects=side_effects"
    )


def test_first_with_subgraph_that_raises():
    inner = composers.make_compose(_node2, _node_that_raises)
    edges = graph.connect_default_terminal(composers.make_first(inner, _node1))
    result = run.to_callable(edges, frozenset([base_types.SkipComputationError]))(
        arg1=_ROOT_VALUE
    )
    assert result.result[graph.DEFAULT_TERMINAL][0] == "node1(root)"


def test_or_with_sink_that_raises():
    edges = graph.connect_default_terminal(
        composers.make_or(
            (_node_that_raises, _node1), merge_fn=_merger_that_raises_when_empty
        )
    )
    result = run.to_callable(edges, frozenset([base_types.SkipComputationError]))(
        arg1=_ROOT_VALUE
    )
    assert result.result[graph.DEFAULT_TERMINAL][0] == "[node1(root)]"


def test_two_terminals():
    """graph = node1 --> node2 --> DEFAULT_TERMINAL, node1 --> TERMINAL2"""
    edges = graph.connect_default_terminal(composers.make_compose(_node2, _node1))
    terminal2 = graph.make_terminal("TERMINAL2", gamla.wrap_tuple)
    edges += (graph.make_standard_edge(source=_node1, destination=terminal2),)
    result = run.to_callable(edges, frozenset([base_types.SkipComputationError]))(
        arg1=_ROOT_VALUE
    )
    assert result.result[graph.DEFAULT_TERMINAL][0] == "node2(node1(root))"
    assert result.result[terminal2][0] == "node1(root)"


def test_two_paths_succeed():
    edges = graph.connect_default_terminal(composers.make_first(_node2, _node1))
    terminal2 = graph.make_terminal("TERMINAL2", gamla.wrap_tuple)
    edges += (graph.make_standard_edge(source=_node1, destination=terminal2),)
    result = run.to_callable(edges, frozenset([base_types.SkipComputationError]))(
        arg1=_ROOT_VALUE
    )

    assert result.result[graph.DEFAULT_TERMINAL][0] == "node2(root)"
    assert result.result[terminal2][0] == "node1(root)"


def test_double_star_signature_considered_unary():
    sink = gamla.juxt(
        lambda some_argname: some_argname + 1,
        lambda different_argname: different_argname * 2,
    )
    assert graph_runners.nullary(composers.make_compose(sink, lambda: 3), sink) == (
        4,
        6,
    )


def test_type_safety_messages(caplog):
    def f(x) -> int:  # Bad typing!
        return "hello " + x

    assert (
        graph_runners.nullary(composers.make_compose(f, lambda: "world"), f)
        == "hello world"
    )
    assert "TypeError" in caplog.text


def test_type_safety_messages_no_overtrigger(caplog):
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
    x = graph.make_source()
    y = graph.make_source()
    graph_runners.variadic_with_state_and_expectations(
        base_types.merge_graphs(
            composers.compose_source_unary(_plus_1, y),
            composers.compose_source_unary(_times_2, x),
            composers.compose_source(_multiply, a, "a"),
            composers.make_compose_future(
                _multiply,
                composers.make_and([_plus_1, _times_2, _multiply], merge_fn=_sum),
                "b",
                None,
            ),
        ),
        _sum,
    )(([[{a: 2, x: 2, y: 2}, 9], [{a: 2, x: 2, y: 2}, 25]]))
