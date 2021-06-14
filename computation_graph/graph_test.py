import asyncio
import functools
import json

import gamla
import pytest

from computation_graph import base_types, composers, graph, run

pytestmark = pytest.mark.asyncio


_ROOT_VALUE = "root"


class GraphTestException(Exception):
    pass


def node1(arg1):
    return f"node1({arg1})"


async def node1_async(arg1):
    await asyncio.sleep(0.1)
    return f"node1({arg1})"


def node2(arg1):
    return f"node2({arg1})"


def node3(arg1, arg2):
    return f"node3(arg1={arg1}, arg2={arg2})"


def node4(y, z=5):
    return f"node4({y}, z={z})"


def node_with_side_effect(arg1, side_effects, state):
    if state is None:
        state = 0
    return base_types.ComputationResult(
        result=f"node_with_side_effect(arg1={arg1},side_effects={side_effects}, state={state + 1})",
        state=(state + 1),
    )


@gamla.curry
def curried_node(arg1, arg2):
    return f"curried_node(arg1={arg1}, arg2={arg2})"


@gamla.curry
def curried_stateful_node(arg1, arg2, state):
    if state is None:
        state = 0

    return base_types.ComputationResult(
        result=f"curried_stateful_node(arg1={arg1}, arg2={arg2}, state={state + 1})",
        state=(state + 1),
    )


def unactionable_node(arg1):
    raise GraphTestException


def merger(args, side_effects):
    return "[" + ",".join(args) + f"], side_effects={side_effects}"


def merger_that_raises_when_empty(args):
    if not args:
        raise GraphTestException
    return "[" + ",".join(args) + "]"


def node_with_optional_param(optional_param: int = 5):
    return f"node_with_optional_param(optional_param={optional_param})"


def reducer_node(arg1, state):
    if state is None:
        state = 0
    return base_types.ComputationResult(
        result=arg1 + f" state={state + 1}", state=state + 1
    )


def sometimes_unactionable_reducer_node(arg1, state):
    if arg1 == "fail":
        raise GraphTestException

    if state is None:
        state = 0

    return base_types.ComputationResult(
        result=arg1 + f" state={state + 1}", state=state + 1
    )


def test_simple():
    cg = run.to_callable(
        graph.connect_default_terminal(
            (graph.make_edge(source=node1, destination=node2, key="arg1"),)
        ),
        frozenset([GraphTestException]),
    )
    result = cg(arg1=_ROOT_VALUE)
    assert isinstance(result, base_types.ComputationResult)
    assert result.result[graph.DEFAULT_TERMINAL][0] == f"node2(node1({_ROOT_VALUE}))"


def test_none_as_input():
    cg = run.to_callable(
        graph.connect_default_terminal(
            (graph.make_edge(source=node1, destination=node2, key="arg1"),)
        ),
        frozenset([GraphTestException]),
    )
    result = cg(arg1=None)

    assert isinstance(result, base_types.ComputationResult)
    assert result.result[graph.DEFAULT_TERMINAL][0] == "node2(node1(None))"


async def test_simple_async():
    cg = run.to_callable(
        graph.connect_default_terminal(
            (graph.make_edge(source=node1_async, destination=node2, key="arg1"),)
        ),
        frozenset([GraphTestException]),
    )
    result = await cg(arg1=_ROOT_VALUE)

    assert isinstance(result, base_types.ComputationResult)
    assert result.result[graph.DEFAULT_TERMINAL][0] == f"node2(node1({_ROOT_VALUE}))"


def test_kwargs():
    cg = run.to_callable(
        graph.connect_default_terminal(
            (
                graph.make_edge(source=node1, destination=node3, key="arg1"),
                graph.make_edge(source=node2, destination=node3, key="arg2"),
            )
        ),
        frozenset([GraphTestException]),
    )

    result = cg(arg1=_ROOT_VALUE)

    assert isinstance(result, base_types.ComputationResult)
    assert (
        result.result[graph.DEFAULT_TERMINAL][0]
        == f"node3(arg1=node1({_ROOT_VALUE}), arg2=node2({_ROOT_VALUE}))"
    )


def test_do_not_allow_kwargs():
    with pytest.raises(AssertionError):
        composers.make_first(lambda **kwargs: 0)


def test_state():
    edges = graph.connect_default_terminal(
        (
            graph.make_edge(source=node1, destination=reducer_node, key="arg1"),
            graph.make_edge(source=reducer_node, destination=node2, key="arg1"),
        )
    )

    cg = run.to_callable(edges, frozenset([GraphTestException]))

    result = cg(arg1=_ROOT_VALUE)
    result = cg(arg1=_ROOT_VALUE, state=result.state)
    result = cg(arg1=_ROOT_VALUE, state=result.state)

    assert isinstance(result, base_types.ComputationResult)
    assert (
        dict(result.state)[
            graph.edges_to_node_id_map(edges)[graph.make_computation_node(reducer_node)]
        ]
        == 3
    )


def test_multiple_inputs():
    edges = graph.connect_default_terminal(
        (
            graph.make_edge(source=node1, destination=node2, key="arg1"),
            graph.make_edge(source=node1, destination=node3, key="arg1"),
            graph.make_edge(source=node2, destination=node3, key="arg2"),
        )
    )
    result = run.to_callable(edges, frozenset([GraphTestException]))(arg1=_ROOT_VALUE)
    assert (
        result.result[graph.DEFAULT_TERMINAL][0]
        == "node3(arg1=node1(root), arg2=node2(node1(root)))"
    )


def test_empty_result():
    edges = graph.connect_default_terminal(
        (graph.make_edge(source=node1, destination=unactionable_node, key="arg1"),)
    )
    result = run.to_callable(edges, frozenset([GraphTestException]))(arg1=_ROOT_VALUE)
    assert not result.result


def test_external_input_and_state():
    edges = graph.connect_default_terminal(
        (
            graph.make_edge(source=node1, destination=node2, key="arg1"),
            graph.make_edge(
                source=node2, destination=node_with_side_effect, key="arg1"
            ),
        )
    )

    cg = run.to_callable(edges, frozenset([GraphTestException]))

    result = cg(arg1=_ROOT_VALUE, side_effects="side_effects")
    result = cg(arg1=_ROOT_VALUE, side_effects="side_effects", state=result.state)
    result = cg(arg1=_ROOT_VALUE, side_effects="side_effects", state=result.state)

    assert (
        result.result[graph.DEFAULT_TERMINAL][0]
        == "node_with_side_effect(arg1=node2(node1(root)),side_effects=side_effects, state=3)"
    )


def test_tuple_source_node():
    edges = graph.connect_default_terminal(
        (
            graph.make_edge(
                source=(node1, node2),
                destination=lambda *args, side_effects: "["
                + ",".join(args)
                + f"], side_effects={side_effects}",
            ),
        )
    )

    result = run.to_callable(edges, frozenset([GraphTestException]))(
        arg1=_ROOT_VALUE, side_effects="side_effects"
    )

    assert (
        result.result[graph.DEFAULT_TERMINAL][0]
        == f"[node1({_ROOT_VALUE}),node2({_ROOT_VALUE})], side_effects=side_effects"
    )


def test_optional():
    edges = graph.connect_default_terminal(
        composers.make_optional(unactionable_node, default_value=None)
    )

    result = run.to_callable(edges, frozenset([GraphTestException]))(arg1=_ROOT_VALUE)

    assert result.result[graph.DEFAULT_TERMINAL][0] is None


def test_optional_with_state():
    edges = graph.connect_default_terminal(
        composers.make_optional(reducer_node, default_value=None)
    )

    result = run.to_callable(edges, frozenset([GraphTestException]))(arg1=_ROOT_VALUE)
    result = run.to_callable(edges, frozenset([GraphTestException]))(
        arg1=_ROOT_VALUE, state=result.state
    )
    result = run.to_callable(edges, frozenset([GraphTestException]))(
        arg1=_ROOT_VALUE, state=result.state
    )

    assert result.result[graph.DEFAULT_TERMINAL][0] == "root state=3"


def test_optional_default_value():
    edges = graph.connect_default_terminal(
        composers.make_optional(unactionable_node, default_value="optional failed")
    )

    result = run.to_callable(edges, frozenset([GraphTestException]))(arg1=_ROOT_VALUE)

    assert result.result[graph.DEFAULT_TERMINAL][0] == "optional failed"


def test_first():
    cg = run.to_callable(
        graph.connect_default_terminal(
            composers.make_first(unactionable_node, node2, node1)
        ),
        frozenset([GraphTestException]),
    )

    result = cg(arg1=_ROOT_VALUE)
    assert result.result[graph.DEFAULT_TERMINAL][0] == "node2(root)"


def test_first_all_unactionable():
    cg = run.to_callable(
        graph.connect_default_terminal(composers.make_first(unactionable_node)),
        frozenset([GraphTestException]),
    )
    result = cg(arg1=_ROOT_VALUE)
    assert not result.result


def test_first_with_state():
    cg = run.to_callable(
        graph.connect_default_terminal(
            composers.make_first(unactionable_node, reducer_node, node1)
        ),
        frozenset([GraphTestException]),
    )

    result = cg(arg1=_ROOT_VALUE)
    result = cg(arg1=_ROOT_VALUE, state=result.state)
    result = cg(arg1=_ROOT_VALUE, state=result.state)

    assert result.result[graph.DEFAULT_TERMINAL][0] == "root state=3"


def test_and():
    edges = graph.connect_default_terminal(
        composers.make_and(funcs=(reducer_node, node2, node1), merge_fn=merger)
    )
    cg = run.to_callable(edges, frozenset([GraphTestException]))
    result = cg(arg1=_ROOT_VALUE, side_effects="side_effects")
    result = cg(arg1=_ROOT_VALUE, state=result.state, side_effects="side_effects")
    result = cg(arg1=_ROOT_VALUE, state=result.state, side_effects="side_effects")

    assert (
        result.result[graph.DEFAULT_TERMINAL][0]
        == "[root state=3,node2(root),node1(root)], side_effects=side_effects"
    )


def test_first_with_and():
    edges = graph.connect_default_terminal(
        composers.make_first(
            unactionable_node,
            composers.make_and(
                (node1, node2),
                merge_fn=functools.partial(merger, side_effects="side_effects"),
            ),
        )
    )

    result = run.to_callable(edges, frozenset([GraphTestException]))(arg1=_ROOT_VALUE)

    assert (
        result.result[graph.DEFAULT_TERMINAL][0]
        == "[node1(root),node2(root)], side_effects=side_effects"
    )


def test_and_with_unactionable():
    edges = graph.connect_default_terminal(
        composers.make_and(
            funcs=(reducer_node, node2, node1, unactionable_node), merge_fn=merger
        )
    )
    result = run.to_callable(edges, frozenset([GraphTestException]))(arg1=_ROOT_VALUE)
    assert not result.result


def test_or():
    edges = graph.connect_default_terminal(
        composers.make_or(
            funcs=(reducer_node, node2, node1, unactionable_node), merge_fn=merger
        )
    )

    result = run.to_callable(edges, frozenset([GraphTestException]))(
        arg1=_ROOT_VALUE, side_effects="side_effects"
    )
    result = run.to_callable(edges, frozenset([GraphTestException]))(
        arg1=_ROOT_VALUE, state=result.state, side_effects="side_effects"
    )
    result = run.to_callable(edges, frozenset([GraphTestException]))(
        arg1=_ROOT_VALUE, state=result.state, side_effects="side_effects"
    )

    assert (
        result.result[graph.DEFAULT_TERMINAL][0]
        == "[root state=3,node2(root),node1(root)], side_effects=side_effects"
    )


def test_graph_wrapping():
    edges = graph.connect_default_terminal(
        composers.make_first(
            unactionable_node,
            composers.make_and(funcs=(node1, node2, node3), merge_fn=merger),
            node1,
        )
    )

    result = run.to_callable(edges, frozenset([GraphTestException]))(
        arg1=_ROOT_VALUE, arg2="node3value", side_effects="side_effects"
    )

    assert (
        result.result[graph.DEFAULT_TERMINAL][0]
        == "[node1(root),node2(root),node3(arg1=root, arg2=node3value)], side_effects=side_effects"
    )


def test_node_with_optional_param():
    edges = graph.connect_default_terminal(
        (
            graph.make_edge(
                source=node_with_optional_param, destination=node1, key="arg1"
            ),
        )
    )

    result = run.to_callable(edges, frozenset([GraphTestException]))()

    assert (
        result.result[graph.DEFAULT_TERMINAL][0]
        == "node1(node_with_optional_param(optional_param=5))"
    )


def test_node_with_bound_optional_param():
    edges = graph.connect_default_terminal(
        (
            graph.make_edge(
                source=node_with_optional_param, destination=node1, key="arg1"
            ),
        )
    )

    result = run.to_callable(edges, frozenset([GraphTestException]))(optional_param=10)

    assert (
        result.result[graph.DEFAULT_TERMINAL][0]
        == "node1(node_with_optional_param(optional_param=10))"
    )


def test_compose():
    result = run.to_callable(
        graph.connect_default_terminal(composers.make_compose(node1, node2)),
        frozenset([GraphTestException]),
    )(arg1=_ROOT_VALUE)

    assert result.result[graph.DEFAULT_TERMINAL][0] == "node1(node2(root))"


def test_compose_with_state():
    cg = run.to_callable(
        graph.connect_default_terminal(
            composers.make_compose(reducer_node, node1, node2)
        ),
        frozenset([GraphTestException]),
    )

    result = cg(arg1=_ROOT_VALUE)
    result = cg(arg1=_ROOT_VALUE, state=result.state)
    result = cg(arg1=_ROOT_VALUE, state=result.state)

    assert result.result[graph.DEFAULT_TERMINAL][0] == "node1(node2(root)) state=3"


def test_compose_with_partial():
    cg = run.to_callable(
        graph.connect_default_terminal(
            composers.make_compose(
                functools.partial(node3, arg2="arg2_partial"), node1, node2
            )
        ),
        frozenset([GraphTestException]),
    )

    result = cg(arg1=_ROOT_VALUE)

    assert (
        result.result[graph.DEFAULT_TERMINAL][0]
        == f"node3(arg1=node1(node2({_ROOT_VALUE})), arg2=arg2_partial)"
    )


def test_partial():
    edges = graph.connect_default_terminal(
        composers.make_first(functools.partial(node3, arg2="arg2_partial"))
    )

    result = run.to_callable(edges, frozenset([GraphTestException]))(arg1=_ROOT_VALUE)

    assert (
        result.result[graph.DEFAULT_TERMINAL][0]
        == "node3(arg1=root, arg2=arg2_partial)"
    )


def test_curry():
    edges = graph.connect_default_terminal(
        composers.make_first(curried_node(arg2="arg2_curried"))
    )

    result = run.to_callable(edges, frozenset([GraphTestException]))(arg1=_ROOT_VALUE)

    assert (
        result.result[graph.DEFAULT_TERMINAL][0]
        == "curried_node(arg1=root, arg2=arg2_curried)"
    )


def test_compose_when_all_arguments_have_a_default():
    edges = graph.connect_default_terminal(
        composers.make_compose(node_with_optional_param, node1)
    )

    result = run.to_callable(edges, frozenset([GraphTestException]))(arg1=_ROOT_VALUE)

    assert (
        result.result[graph.DEFAULT_TERMINAL][0]
        == "node_with_optional_param(optional_param=node1(root))"
    )


def test_optional_with_sometimes_unactionable_reducer():
    edges = graph.connect_default_terminal(
        composers.make_optional(sometimes_unactionable_reducer_node, default_value=None)
    )
    cg = run.to_callable(edges, frozenset([GraphTestException]))
    result = cg(arg1=_ROOT_VALUE)
    result = cg(arg1="fail", state=result.state)
    result = cg(arg1=_ROOT_VALUE, state=result.state)

    assert result.result[graph.DEFAULT_TERMINAL][0] == "root state=2"


def test_unary_graph_composition():
    inner = composers.make_compose(node1, node4)
    edges = graph.connect_default_terminal(composers.make_first(inner))
    result = run.to_callable(edges, frozenset([GraphTestException]))(
        y=_ROOT_VALUE, z=10
    )

    assert result.result[graph.DEFAULT_TERMINAL][0] == "node1(node4(root, z=10))"


def test_curry_with_state():

    edges = graph.connect_default_terminal(composers.make_first(curried_stateful_node))

    cg = run.to_callable(edges, frozenset([GraphTestException]))
    result = cg(arg1=_ROOT_VALUE, arg2="arg2")
    result = cg(arg1=_ROOT_VALUE, arg2="arg2", state=result.state)
    result = cg(arg1=_ROOT_VALUE, arg2="arg2", state=result.state)

    assert (
        result.result[graph.DEFAULT_TERMINAL][0]
        == "curried_stateful_node(arg1=root, arg2=arg2, state=3)"
    )


def test_state_is_serializable():
    edges = graph.connect_default_terminal(
        (
            graph.make_edge(source=node1, destination=reducer_node, key="arg1"),
            graph.make_edge(source=reducer_node, destination=node2, key="arg1"),
        )
    )

    cg = run.to_callable(edges, frozenset([GraphTestException]))
    result = cg(arg1=_ROOT_VALUE)
    result = cg(arg1=_ROOT_VALUE, state=result.state)
    result = cg(arg1=_ROOT_VALUE, state=result.state)

    json.dumps(result.state)


def test_compose_compose():

    inner_graph = composers.make_compose(curried_node, node1, node3, node2, key="arg1")
    assert len(inner_graph) == 3

    edges = graph.connect_default_terminal(
        composers.make_compose(inner_graph, node4, key="arg2")
    )

    result = run.to_callable(edges, frozenset([GraphTestException]))(
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
            composers.make_first(unactionable_node, node1, node2), node3, key="arg1"
        )
    )
    result = run.to_callable(edges, frozenset([GraphTestException]))(
        arg1="arg1", arg2="arg2"
    )
    assert (
        result.result[graph.DEFAULT_TERMINAL][0] == "node1(node3(arg1=arg1, arg2=arg2))"
    )


def test_first_after_compose():
    inner_edges = composers.make_compose(node1, node2)

    cg = run.to_callable(
        graph.connect_default_terminal(
            composers.make_first(unactionable_node, inner_edges, node1)
        ),
        frozenset([GraphTestException]),
    )

    result = cg(arg1="arg1")
    assert result.result[graph.DEFAULT_TERMINAL][0] == "node1(node2(arg1))"


def test_first_first():
    inner_first = composers.make_first(unactionable_node, node1, node2)

    cg = run.to_callable(
        graph.connect_default_terminal(
            composers.make_first(unactionable_node, inner_first, node1)
        ),
        frozenset([GraphTestException]),
    )

    result = cg(arg1=_ROOT_VALUE)

    assert result.result[graph.DEFAULT_TERMINAL][0] == f"node1({_ROOT_VALUE})"


def test_compose_with_node_already_in_graph():
    inner_edges1 = composers.make_and((node2, node1), merge_fn=merger)
    inner_edges2 = composers.make_compose(node3, node1, key="arg1")
    edges = graph.connect_default_terminal(
        composers.make_compose(inner_edges1, inner_edges2, key="arg1")
    )

    result = run.to_callable(edges, frozenset([GraphTestException]))(
        arg1=_ROOT_VALUE, arg2="arg2", side_effects="side_effects"
    )

    assert (
        result.result[graph.DEFAULT_TERMINAL][0]
        == "[node2(node3(arg1=node1(root), arg2=arg2)),node1(root)], side_effects=side_effects"
    )


def test_first_with_subgraph_that_raises():
    inner = composers.make_compose(node2, unactionable_node)
    edges = graph.connect_default_terminal(composers.make_first(inner, node1))
    result = run.to_callable(edges, frozenset([GraphTestException]))(arg1=_ROOT_VALUE)
    assert result.result[graph.DEFAULT_TERMINAL][0] == "node1(root)"


def test_or_with_sink_that_raises():
    edges = graph.connect_default_terminal(
        composers.make_or(
            (unactionable_node, node1), merge_fn=merger_that_raises_when_empty
        )
    )
    result = run.to_callable(edges, frozenset([GraphTestException]))(arg1=_ROOT_VALUE)

    assert result.result[graph.DEFAULT_TERMINAL][0] == "[node1(root)]"


def test_two_terminals():
    """ graph = node1 --> node2 --> DEFAULT_TERMINAL, node1 --> TERMINAL2"""
    edges = graph.connect_default_terminal(composers.make_compose(node2, node1))
    terminal2 = graph.make_terminal("TERMINAL2", gamla.wrap_tuple)
    edges += (graph.make_edge(source=node1, destination=terminal2),)

    result = run.to_callable(edges, frozenset([GraphTestException]))(arg1=_ROOT_VALUE)
    assert result.result[graph.DEFAULT_TERMINAL][0] == "node2(node1(root))"
    assert result.result[terminal2][0] == "node1(root)"


def test_two_paths_succeed():
    edges = graph.connect_default_terminal(composers.make_first(node2, node1))
    terminal2 = graph.make_terminal("TERMINAL2", gamla.wrap_tuple)
    edges += (graph.make_edge(source=node1, destination=terminal2),)
    result = run.to_callable(edges, frozenset([GraphTestException]))(arg1=_ROOT_VALUE)

    assert result.result[graph.DEFAULT_TERMINAL][0] == "node2(root)"
    assert result.result[terminal2][0] == "node1(root)"
