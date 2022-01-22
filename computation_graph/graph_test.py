import asyncio
import functools
import json
from typing import Callable

import gamla
import pytest

from computation_graph import base_types, composers, graph, legacy, run

pytestmark = pytest.mark.asyncio


_ROOT_VALUE = "root"


class _GraphTestError(Exception):
    pass


def _node1(arg1):
    return f"node1({arg1})"


async def _node1_async(arg1):
    await asyncio.sleep(0.1)
    return f"node1({arg1})"


def _node2(arg1):
    return f"node2({arg1})"


def _node3(arg1, arg2):
    return f"node3(arg1={arg1}, arg2={arg2})"


def _node4(y, z=5):
    return f"node4({y}, z={z})"


def _node_with_side_effect(arg1, side_effects, cur_int):
    return f"node_with_side_effect(arg1={arg1},side_effects={side_effects}, cur_int={cur_int + 1})"


@gamla.curry
def _curried_node(arg1, arg2):
    return f"curried_node(arg1={arg1}, arg2={arg2})"


def _unactionable_node(arg1):
    raise _GraphTestError


def _merger(args, side_effects):
    return "[" + ",".join(args) + f"], side_effects={side_effects}"


def _merger_that_raises_when_empty(args):
    if not args:
        raise _GraphTestError
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
        raise _GraphTestError
    return arg1 + f" state={cur_int + 1}"


def _unary_graph(g: base_types.GraphType, source: Callable, sink: Callable) -> Callable:
    real_source = graph.make_source()
    return gamla.compose(
        gamla.itemgetter(graph.make_computation_node(sink)),
        run.to_callable_strict(
            base_types.merge_graphs(
                g, composers.compose_left_future(real_source, source, None, None)
            )
        ),
        gamla.wrap_dict(real_source),
    )


def _unary_graph_with_state(
    g: base_types.GraphType, source: Callable, sink: Callable
) -> Callable:
    real_source = graph.make_source()
    f = run.to_callable_strict(
        base_types.merge_graphs(
            g, composers.compose_left_future(real_source, source, None, None)
        )
    )

    def inner(*turns):
        prev = {}
        for turn in turns:
            prev = f({real_source: turn, **prev})
        return prev[graph.make_computation_node(sink)]

    return inner


def _unary_graph_with_state_and_expectations(
    g: base_types.GraphType, source: Callable, sink: Callable
) -> Callable:
    real_source = graph.make_source()
    return gamla.compose(
        _variadic_with_state_and_expectations(
            base_types.merge_graphs(
                g, composers.compose_left_future(real_source, source, None, None)
            ),
            sink,
        ),
        tuple,
        gamla.map(lambda t: ({real_source: t[0]}, t[1])),
    )


def _variadic_with_state_and_expectations(g, sink):
    f = run.to_callable_strict(g)

    def inner(turns):
        prev = {}
        for turn, expectation in turns:
            prev = f({**turn, **prev})
            assert prev[graph.make_computation_node(sink)] == expectation

    return inner


def _nullary_graph(g, sink):
    return gamla.compose(
        gamla.itemgetter(graph.make_computation_node(sink)),
        run.to_callable_strict(g),
        gamla.just({}),
    )()


def test_simple():
    for v in ["root", None]:
        assert (
            _unary_graph(composers.compose_left_unary(_node1, _node2), _node1, _node2)(
                v
            )
            == f"node2(node1({v}))"
        )


async def test_simple_async():
    assert (
        await _unary_graph(
            composers.compose_left_unary(_node1_async, _node2), _node1_async, _node2
        )("hi")
        == "node2(node1(hi))"
    )


def test_kwargs():
    def source(x):
        return x

    assert (
        _unary_graph(
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
    f = _unary_graph_with_state(
        base_types.merge_graphs(
            composers.compose_left(_node1, _node_with_state_as_arg, key="arg1"),
            composers.compose_left_unary(_node_with_state_as_arg, _node2),
        ),
        _node1,
        _node2,
    )
    assert f(_ROOT_VALUE, _ROOT_VALUE, _ROOT_VALUE) == "node2(node1(root) state=3)"


def test_self_future_edge():
    f = _unary_graph_with_state(
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


def test_multiple_inputs():
    edges = graph.connect_default_terminal(
        (
            graph.make_standard_edge(source=_node1, destination=_node2, key="arg1"),
            graph.make_standard_edge(source=_node1, destination=_node3, key="arg1"),
            graph.make_standard_edge(source=_node2, destination=_node3, key="arg2"),
        )
    )
    result = run.to_callable(edges, frozenset([_GraphTestError]))(arg1=_ROOT_VALUE)
    assert (
        result.result[graph.DEFAULT_TERMINAL][0]
        == "node3(arg1=node1(root), arg2=node2(node1(root)))"
    )


def test_empty_result():
    edges = graph.connect_default_terminal(
        (
            graph.make_standard_edge(
                source=_node1, destination=_unactionable_node, key="arg1"
            ),
        )
    )
    assert run.to_callable(edges, frozenset([_GraphTestError]))(
        arg1=_ROOT_VALUE
    ).result == {graph.DEFAULT_TERMINAL: {}}


def test_external_input_and_future_edge():
    edges = graph.connect_default_terminal(
        (
            graph.make_standard_edge(source=_node1, destination=_node2, key="arg1"),
            graph.make_standard_edge(
                source=_node2, destination=_node_with_side_effect, key="arg1"
            ),
            graph.make_standard_edge(
                source=_next_int, destination=_node_with_side_effect, key="cur_int"
            ),
            graph.make_future_edge(source=_next_int, destination=_next_int),
        )
    )

    cg = run.to_callable(edges, frozenset([_GraphTestError]))

    result = cg(arg1=_ROOT_VALUE, side_effects="side_effects")
    result = cg(arg1=_ROOT_VALUE, side_effects="side_effects", state=result.state)
    result = cg(arg1=_ROOT_VALUE, side_effects="side_effects", state=result.state)

    assert (
        result.result[graph.DEFAULT_TERMINAL][0]
        == "node_with_side_effect(arg1=node2(node1(root)),side_effects=side_effects, cur_int=3)"
    )


def test_tuple_source_node():
    edges = graph.connect_default_terminal(
        (
            graph.make_standard_edge(
                source=(_node1, _node2),
                destination=lambda *args, side_effects: "["
                + ",".join(args)
                + f"], side_effects={side_effects}",
            ),
        )
    )

    result = run.to_callable(edges, frozenset([_GraphTestError]))(
        arg1=_ROOT_VALUE, side_effects="side_effects"
    )

    assert (
        result.result[graph.DEFAULT_TERMINAL][0]
        == f"[node1({_ROOT_VALUE}),node2({_ROOT_VALUE})], side_effects=side_effects"
    )


def test_optional():
    edges = graph.connect_default_terminal(
        composers.make_optional(_unactionable_node, default_value=None)
    )

    result = run.to_callable(edges, frozenset([_GraphTestError]))(arg1=_ROOT_VALUE)

    assert result.result[graph.DEFAULT_TERMINAL][0] is None


def test_optional_with_future_edge():
    edges = graph.connect_default_terminal(
        composers.make_optional(_reducer_node, default_value=None)
        + (
            graph.make_standard_edge(
                source=_next_int, destination=_reducer_node, key="cur_int"
            ),
            graph.make_future_edge(source=_next_int, destination=_next_int),
        )
    )
    cg = run.to_callable(edges, frozenset([_GraphTestError]))

    result = cg(arg1=_ROOT_VALUE)
    result = cg(arg1=_ROOT_VALUE, state=result.state)
    result = cg(arg1=_ROOT_VALUE, state=result.state)
    assert result.result[graph.DEFAULT_TERMINAL][0] == "root cur_int=3"


def test_optional_default_value():
    edges = graph.connect_default_terminal(
        composers.make_optional(_unactionable_node, default_value="optional failed")
    )

    result = run.to_callable(edges, frozenset([_GraphTestError]))(arg1=_ROOT_VALUE)

    assert result.result[graph.DEFAULT_TERMINAL][0] == "optional failed"


def test_first():
    cg = run.to_callable(
        graph.connect_default_terminal(
            composers.make_first(_unactionable_node, _node2, _node1)
        ),
        frozenset([_GraphTestError]),
    )

    result = cg(arg1=_ROOT_VALUE)
    assert result.result[graph.DEFAULT_TERMINAL][0] == "node2(root)"


def test_first_all_unactionable():
    cg = run.to_callable(
        graph.connect_default_terminal(composers.make_first(_unactionable_node)),
        frozenset([_GraphTestError]),
    )
    result = cg(arg1=_ROOT_VALUE)
    assert result.result == {graph.DEFAULT_TERMINAL: {}}


def test_first_with_future_edge():
    cg = run.to_callable(
        graph.connect_default_terminal(
            composers.make_first(_unactionable_node, _reducer_node, _node1)
            + (
                graph.make_standard_edge(
                    source=_next_int, destination=_reducer_node, key="cur_int"
                ),
                graph.make_future_edge(source=_next_int, destination=_next_int),
            )
        ),
        frozenset([_GraphTestError]),
    )

    result = cg(arg1=_ROOT_VALUE)
    result = cg(arg1=_ROOT_VALUE, state=result.state)
    result = cg(arg1=_ROOT_VALUE, state=result.state)
    assert result.result[graph.DEFAULT_TERMINAL][0] == "root cur_int=3"


def test_and():
    edges = graph.connect_default_terminal(
        composers.make_and(funcs=(_reducer_node, _node2, _node1), merge_fn=_merger)
        + (
            graph.make_standard_edge(
                source=_next_int, destination=_reducer_node, key="cur_int"
            ),
            graph.make_future_edge(source=_next_int, destination=_next_int),
        )
    )
    cg = run.to_callable(edges, frozenset([_GraphTestError]))
    result = cg(arg1=_ROOT_VALUE, side_effects="side_effects")
    result = cg(arg1=_ROOT_VALUE, state=result.state, side_effects="side_effects")
    result = cg(arg1=_ROOT_VALUE, state=result.state, side_effects="side_effects")
    assert (
        result.result[graph.DEFAULT_TERMINAL][0]
        == "[root cur_int=3,node2(root),node1(root)], side_effects=side_effects"
    )


def test_first_with_and():
    edges = graph.connect_default_terminal(
        composers.make_first(
            _unactionable_node,
            composers.make_and(
                (_node1, _node2),
                merge_fn=functools.partial(_merger, side_effects="side_effects"),
            ),
        )
    )

    result = run.to_callable(edges, frozenset([_GraphTestError]))(arg1=_ROOT_VALUE)

    assert (
        result.result[graph.DEFAULT_TERMINAL][0]
        == "[node1(root),node2(root)], side_effects=side_effects"
    )


def test_and_with_unactionable():
    edges = graph.connect_default_terminal(
        composers.make_and(
            funcs=(_reducer_node, _node2, _node1, _unactionable_node), merge_fn=_merger
        )
        + (
            graph.make_standard_edge(
                source=_next_int, destination=_reducer_node, key="cur_int"
            ),
            graph.make_future_edge(source=_next_int, destination=_next_int),
        )
    )
    assert run.to_callable(edges, frozenset([_GraphTestError]))(
        arg1=_ROOT_VALUE
    ).result == {graph.DEFAULT_TERMINAL: {}}


def test_or():
    edges = graph.connect_default_terminal(
        composers.make_or(
            funcs=(_reducer_node, _node2, _node1, _unactionable_node), merge_fn=_merger
        )
        + (
            graph.make_standard_edge(
                source=_next_int, destination=_reducer_node, key="cur_int"
            ),
            graph.make_future_edge(source=_next_int, destination=_next_int),
        )
    )

    result = run.to_callable(edges, frozenset([_GraphTestError]))(
        arg1=_ROOT_VALUE, side_effects="side_effects"
    )
    result = run.to_callable(edges, frozenset([_GraphTestError]))(
        arg1=_ROOT_VALUE, state=result.state, side_effects="side_effects"
    )
    result = run.to_callable(edges, frozenset([_GraphTestError]))(
        arg1=_ROOT_VALUE, state=result.state, side_effects="side_effects"
    )

    assert (
        result.result[graph.DEFAULT_TERMINAL][0]
        == "[root cur_int=3,node2(root),node1(root)], side_effects=side_effects"
    )


def test_graph_wrapping():
    edges = graph.connect_default_terminal(
        composers.make_first(
            _unactionable_node,
            composers.make_and(funcs=(_node1, _node2, _node3), merge_fn=_merger),
            _node1,
        )
    )

    result = run.to_callable(edges, frozenset([_GraphTestError]))(
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

    result = run.to_callable(edges, frozenset([_GraphTestError]))()

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

    result = run.to_callable(edges, frozenset([_GraphTestError]))(optional_param=10)

    assert (
        result.result[graph.DEFAULT_TERMINAL][0]
        == "node1(node_with_optional_param(optional_param=10))"
    )


def test_compose():
    result = run.to_callable(
        graph.connect_default_terminal(composers.make_compose(_node1, _node2)),
        frozenset([_GraphTestError]),
    )(arg1=_ROOT_VALUE)

    assert result.result[graph.DEFAULT_TERMINAL][0] == "node1(node2(root))"


def test_compose_with_future_edge():
    cg = run.to_callable(
        graph.connect_default_terminal(
            composers.make_compose(_node1, _node2)
            + (
                graph.make_standard_edge(
                    source=_node1, destination=_reducer_node, key="arg1"
                ),
                graph.make_standard_edge(
                    source=_next_int, destination=_reducer_node, key="cur_int"
                ),
                graph.make_future_edge(source=_next_int, destination=_next_int),
            )
        ),
        frozenset([_GraphTestError]),
    )

    result = cg(arg1=_ROOT_VALUE)
    result = cg(arg1=_ROOT_VALUE, state=result.state)
    result = cg(arg1=_ROOT_VALUE, state=result.state)
    assert result.result[graph.DEFAULT_TERMINAL][0] == "node1(node2(root)) cur_int=3"


def test_compose_when_all_arguments_have_a_default():
    edges = graph.connect_default_terminal(
        composers.make_compose(_node_with_optional_param, _node1)
    )

    result = run.to_callable(edges, frozenset([_GraphTestError]))(arg1=_ROOT_VALUE)

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
    cg = run.to_callable(edges, frozenset([_GraphTestError]))
    result = cg(arg1=_ROOT_VALUE)
    result = cg(arg1="fail", state=result.state)
    result = cg(arg1=_ROOT_VALUE, state=result.state)

    assert result.result[graph.DEFAULT_TERMINAL][0] == "root state=2"


def test_unary_graph_composition():
    inner = composers.make_compose(_node1, _node4)
    edges = graph.connect_default_terminal(composers.make_first(inner))
    result = run.to_callable(edges, frozenset([_GraphTestError]))(y=_ROOT_VALUE, z=10)

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

    cg = run.to_callable(edges, frozenset([_GraphTestError]))
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

    result = run.to_callable(edges, frozenset([_GraphTestError]))(
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
            composers.make_first(_unactionable_node, _node1, _node2), _node3, key="arg1"
        )
    )
    result = run.to_callable(edges, frozenset([_GraphTestError]))(
        arg1="arg1", arg2="arg2"
    )
    assert (
        result.result[graph.DEFAULT_TERMINAL][0] == "node1(node3(arg1=arg1, arg2=arg2))"
    )


def test_first_after_compose():
    inner_edges = composers.make_compose(_node1, _node2)

    cg = run.to_callable(
        graph.connect_default_terminal(
            composers.make_first(_unactionable_node, inner_edges, _node1)
        ),
        frozenset([_GraphTestError]),
    )

    result = cg(arg1="arg1")
    assert result.result[graph.DEFAULT_TERMINAL][0] == "node1(node2(arg1))"


def test_first_first():
    inner_first = composers.make_first(_unactionable_node, _node1, _node2)

    cg = run.to_callable(
        graph.connect_default_terminal(
            composers.make_first(_unactionable_node, inner_first, _node1)
        ),
        frozenset([_GraphTestError]),
    )

    result = cg(arg1=_ROOT_VALUE)

    assert result.result[graph.DEFAULT_TERMINAL][0] == f"node1({_ROOT_VALUE})"


def test_compose_with_node_already_in_graph():
    inner_edges1 = composers.make_and((_node2, _node1), merge_fn=_merger)
    inner_edges2 = composers.make_compose(_node3, _node1, key="arg1")
    edges = graph.connect_default_terminal(
        composers.make_compose(inner_edges1, inner_edges2, key="arg1")
    )

    result = run.to_callable(edges, frozenset([_GraphTestError]))(
        arg1=_ROOT_VALUE, arg2="arg2", side_effects="side_effects"
    )

    assert (
        result.result[graph.DEFAULT_TERMINAL][0]
        == "[node2(node3(arg1=node1(root), arg2=arg2)),node1(root)], side_effects=side_effects"
    )


def test_first_with_subgraph_that_raises():
    inner = composers.make_compose(_node2, _unactionable_node)
    edges = graph.connect_default_terminal(composers.make_first(inner, _node1))
    result = run.to_callable(edges, frozenset([_GraphTestError]))(arg1=_ROOT_VALUE)
    assert result.result[graph.DEFAULT_TERMINAL][0] == "node1(root)"


def test_or_with_sink_that_raises():
    edges = graph.connect_default_terminal(
        composers.make_or(
            (_unactionable_node, _node1), merge_fn=_merger_that_raises_when_empty
        )
    )
    result = run.to_callable(edges, frozenset([_GraphTestError]))(arg1=_ROOT_VALUE)
    assert result.result[graph.DEFAULT_TERMINAL][0] == "[node1(root)]"


def test_two_terminals():
    """graph = node1 --> node2 --> DEFAULT_TERMINAL, node1 --> TERMINAL2"""
    edges = graph.connect_default_terminal(composers.make_compose(_node2, _node1))
    terminal2 = graph.make_terminal("TERMINAL2", gamla.wrap_tuple)
    edges += (graph.make_standard_edge(source=_node1, destination=terminal2),)
    result = run.to_callable(edges, frozenset([_GraphTestError]))(arg1=_ROOT_VALUE)
    assert result.result[graph.DEFAULT_TERMINAL][0] == "node2(node1(root))"
    assert result.result[terminal2][0] == "node1(root)"


def test_two_paths_succeed():
    edges = graph.connect_default_terminal(composers.make_first(_node2, _node1))
    terminal2 = graph.make_terminal("TERMINAL2", gamla.wrap_tuple)
    edges += (graph.make_standard_edge(source=_node1, destination=terminal2),)
    result = run.to_callable(edges, frozenset([_GraphTestError]))(arg1=_ROOT_VALUE)

    assert result.result[graph.DEFAULT_TERMINAL][0] == "node2(root)"
    assert result.result[terminal2][0] == "node1(root)"


def test_double_star_signature_considered_unary():
    sink = gamla.juxt(
        lambda some_argname: some_argname + 1,
        lambda different_argname: different_argname * 2,
    )
    assert _nullary_graph(composers.make_compose(sink, lambda: 3), sink) == (4, 6)


def test_type_safety_messages(caplog):
    def f(x) -> int:  # Bad typing!
        return "hello " + x

    assert (
        _nullary_graph(composers.make_compose(f, lambda: "world"), f) == "hello world"
    )
    assert "TypeError" in caplog.text


def test_type_safety_messages_no_overtrigger(caplog):
    def f(x) -> str:
        return "hello " + x

    assert (
        _nullary_graph(composers.make_compose(f, lambda: "world"), f) == "hello world"
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
    _unary_graph_with_state_and_expectations(
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

    f = _unary_graph_with_state(
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

    f = _unary_graph(
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
    _variadic_with_state_and_expectations(
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
