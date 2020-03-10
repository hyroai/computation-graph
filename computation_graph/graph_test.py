import functools
import json
import unittest

import toolz

from computation_graph import base_types, composers, graph, run

_ROOT_VALUE = "root"


class GraphTestException(Exception):
    pass


def node1(arg1):
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


@toolz.curry
def curried_node(arg1, arg2):
    return f"curried_node(arg1={arg1}, arg2={arg2})"


@toolz.curry
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


class TestComputationGraph(unittest.TestCase):
    def test_simple(self):
        cg = run.to_callable(
            (graph.make_edge(source=node1, destination=node2, key="arg1"),)
        )
        result = cg(arg1=_ROOT_VALUE)

        self.assertIsInstance(result, base_types.ComputationResult)
        self.assertEqual(result.result, f"node2(node1({_ROOT_VALUE}))")

    def test_kwargs(self):
        cg = run.to_callable(
            (
                graph.make_edge(source=node1, destination=node3, key="arg1"),
                graph.make_edge(source=node2, destination=node3, key="arg2"),
            )
        )

        result = cg(arg1=_ROOT_VALUE)

        self.assertIsInstance(result, base_types.ComputationResult)
        self.assertEqual(
            result.result,
            f"node3(arg1=node1({_ROOT_VALUE}), arg2=node2({_ROOT_VALUE}))",
        )

    def test_do_not_allow_kwargs(self):

        self.assertRaises(
            AssertionError,
            composers.make_first,
            lambda **kwargs: 0,
            exception_type=GraphTestException,
        )

    def test_state(self):
        edges = (
            graph.make_edge(source=node1, destination=reducer_node, key="arg1"),
            graph.make_edge(source=reducer_node, destination=node2, key="arg1"),
        )

        cg = run.to_callable(edges)

        result = cg(arg1=_ROOT_VALUE)
        result = cg(arg1=_ROOT_VALUE, state=result.state)
        result = cg(arg1=_ROOT_VALUE, state=result.state)

        self.assertIsInstance(result, base_types.ComputationResult)
        self.assertEqual(
            dict(result.state)[
                graph.infer_node_id(edges, graph.make_computation_node(reducer_node))
            ],
            3,
        )

    def test_multiple_inputs(self):
        edges = (
            graph.make_edge(source=node1, destination=node2, key="arg1"),
            graph.make_edge(source=node1, destination=node3, key="arg1"),
            graph.make_edge(source=node2, destination=node3, key="arg2"),
        )

        result = run.to_callable(edges)(arg1=_ROOT_VALUE)

        self.assertEqual(
            result.result, "node3(arg1=node1(root), arg2=node2(node1(root)))"
        )

    def test_exception(self):
        edges = (
            graph.make_edge(source=node1, destination=unactionable_node, key="arg1"),
        )

        self.assertRaises(GraphTestException, run.to_callable(edges), arg1=_ROOT_VALUE)

    def test_external_input_and_state(self):
        edges = (
            graph.make_edge(source=node1, destination=node2, key="arg1"),
            graph.make_edge(
                source=node2, destination=node_with_side_effect, key="arg1"
            ),
        )

        cg = run.to_callable(edges)

        result = cg(arg1=_ROOT_VALUE, side_effects="side_effects")
        result = cg(arg1=_ROOT_VALUE, side_effects="side_effects", state=result.state)
        result = cg(arg1=_ROOT_VALUE, side_effects="side_effects", state=result.state)

        self.assertEqual(
            result.result,
            "node_with_side_effect(arg1=node2(node1(root)),side_effects=side_effects, state=3)",
        )

    def test_tuple_source_node(self):
        edges = (
            graph.make_edge(
                source=(node1, node2),
                destination=lambda *args, side_effects: "["
                + ",".join(args)
                + f"], side_effects={side_effects}",
            ),
        )

        result = run.to_callable(edges)(arg1=_ROOT_VALUE, side_effects="side_effects")

        self.assertEqual(
            result.result,
            f"[node1({_ROOT_VALUE}),node2({_ROOT_VALUE})], side_effects=side_effects",
        )

    def test_optional(self):
        edges = composers.make_optional(
            unactionable_node, exception_type=GraphTestException, default_value=None
        )

        result = run.to_callable(edges)(arg1=_ROOT_VALUE)

        self.assertIsNone(result.result)

    def test_optional_with_state(self):
        edges = composers.make_optional(
            reducer_node, exception_type=GraphTestException, default_value=None
        )

        result = run.to_callable(edges)(arg1=_ROOT_VALUE)
        result = run.to_callable(edges)(arg1=_ROOT_VALUE, state=result.state)
        result = run.to_callable(edges)(arg1=_ROOT_VALUE, state=result.state)

        self.assertEqual(result.result, "root state=3")

    def test_optional_default_value(self):
        edges = composers.make_optional(
            unactionable_node,
            exception_type=GraphTestException,
            default_value="optional failed",
        )

        result = run.to_callable(edges)(arg1=_ROOT_VALUE)

        self.assertEqual(result.result, "optional failed")

    def test_first(self):
        cg = run.to_callable(
            composers.make_first(
                unactionable_node, node2, node1, exception_type=GraphTestException
            )
        )

        result = cg(arg1=_ROOT_VALUE)
        self.assertEqual(result.result, "node2(root)")

    def test_first_all_unactionable(self):
        cg = run.to_callable(
            composers.make_first(unactionable_node, exception_type=GraphTestException)
        )
        self.assertRaises(GraphTestException, cg, arg1=_ROOT_VALUE)

    def test_first_with_state(self):
        cg = run.to_callable(
            composers.make_first(
                unactionable_node,
                reducer_node,
                node1,
                exception_type=GraphTestException,
            )
        )

        result = cg(arg1=_ROOT_VALUE)
        result = cg(arg1=_ROOT_VALUE, state=result.state)
        result = cg(arg1=_ROOT_VALUE, state=result.state)

        self.assertEqual(result.result, "root state=3")

    def test_and(self):
        edges = composers.make_and(funcs=(reducer_node, node2, node1), merge_fn=merger)
        cg = run.to_callable(edges)
        result = cg(arg1=_ROOT_VALUE, side_effects="side_effects")
        result = cg(arg1=_ROOT_VALUE, state=result.state, side_effects="side_effects")
        result = cg(arg1=_ROOT_VALUE, state=result.state, side_effects="side_effects")

        self.assertEqual(
            result.result,
            "[root state=3,node2(root),node1(root)], side_effects=side_effects",
        )

    def test_first_with_and(self):
        edges = composers.make_first(
            unactionable_node,
            composers.make_and(
                (node1, node2),
                merge_fn=functools.partial(merger, side_effects="side_effects"),
            ),
            exception_type=GraphTestException,
        )

        result = run.to_callable(edges)(arg1=_ROOT_VALUE)

        self.assertEqual(
            result.result, "[node1(root),node2(root)], side_effects=side_effects"
        )

    def test_and_with_unactionable(self):
        edges = composers.make_and(
            funcs=(reducer_node, node2, node1, unactionable_node), merge_fn=merger
        )

        self.assertRaises(GraphTestException, run.to_callable(edges), arg1=_ROOT_VALUE)

    def test_or(self):
        edges = composers.make_or(
            funcs=(reducer_node, node2, node1, unactionable_node),
            merge_fn=merger,
            exception_type=GraphTestException,
        )

        result = run.to_callable(edges)(arg1=_ROOT_VALUE, side_effects="side_effects")
        result = run.to_callable(edges)(
            arg1=_ROOT_VALUE, state=result.state, side_effects="side_effects"
        )
        result = run.to_callable(edges)(
            arg1=_ROOT_VALUE, state=result.state, side_effects="side_effects"
        )

        self.assertEqual(
            result.result,
            "[root state=3,node2(root),node1(root)], side_effects=side_effects",
        )

    def test_graph_wrapping(self):
        edges = composers.make_first(
            unactionable_node,
            composers.make_and(funcs=(node1, node2, node3), merge_fn=merger),
            node1,
            exception_type=GraphTestException,
        )

        result = run.to_callable(edges)(
            arg1=_ROOT_VALUE, arg2="node3value", side_effects="side_effects"
        )

        self.assertEqual(
            result.result,
            "[node1(root),node2(root),node3(arg1=root, arg2=node3value)], side_effects=side_effects",
        )

    def test_node_with_optional_param(self):
        edges = (
            graph.make_edge(
                source=node_with_optional_param, destination=node1, key="arg1"
            ),
        )

        result = run.to_callable(edges)()

        self.assertEqual(
            result.result, f"node1(node_with_optional_param(optional_param=5))"
        )

    def test_node_with_bound_optional_param(self):
        edges = (
            graph.make_edge(
                source=node_with_optional_param, destination=node1, key="arg1"
            ),
        )

        result = run.to_callable(edges)(optional_param=10)

        self.assertEqual(
            result.result, f"node1(node_with_optional_param(optional_param=10))"
        )

    def test_compose(self):
        result = run.to_callable(composers.make_compose(node1, node2))(arg1=_ROOT_VALUE)

        self.assertEqual(result.result, "node1(node2(root))")

    def test_compose_with_state(self):
        cg = run.to_callable(composers.make_compose(reducer_node, node1, node2))

        result = cg(arg1=_ROOT_VALUE)
        result = cg(arg1=_ROOT_VALUE, state=result.state)
        result = cg(arg1=_ROOT_VALUE, state=result.state)

        self.assertEqual(result.result, "node1(node2(root)) state=3")

    def test_compose_with_partial(self):
        cg = run.to_callable(
            composers.make_compose(
                functools.partial(node3, arg2="arg2_partial"), node1, node2
            )
        )

        result = cg(arg1=_ROOT_VALUE)

        self.assertEqual(
            result.result, f"node3(arg1=node1(node2({_ROOT_VALUE})), arg2=arg2_partial)"
        )

    def test_partial(self):
        edges = composers.make_first(
            functools.partial(node3, arg2="arg2_partial"),
            exception_type=GraphTestException,
        )

        result = run.to_callable(edges)(arg1=_ROOT_VALUE)

        self.assertEqual(result.result, "node3(arg1=root, arg2=arg2_partial)")

    def test_curry(self):
        edges = composers.make_first(
            curried_node(arg2="arg2_curried"), exception_type=GraphTestException
        )

        result = run.to_callable(edges)(arg1=_ROOT_VALUE)

        self.assertEqual(result.result, "curried_node(arg1=root, arg2=arg2_curried)")

    def test_compose_when_all_arguments_have_a_default(self):
        edges = composers.make_compose(node_with_optional_param, node1)

        result = run.to_callable(edges)(arg1=_ROOT_VALUE)

        self.assertEqual(
            result.result, "node_with_optional_param(optional_param=node1(root))"
        )

    def test_optional_with_sometimes_unactionable_reducer(self):
        edges = composers.make_optional(
            sometimes_unactionable_reducer_node, GraphTestException, default_value=None
        )
        cg = run.to_callable(edges)
        result = cg(arg1=_ROOT_VALUE)
        result = cg(arg1="fail", state=result.state)
        result = cg(arg1=_ROOT_VALUE, state=result.state)

        self.assertEqual(result.result, "root state=2")

    def test_unary_graph_composition(self):

        inner = composers.make_compose(node1, node4)
        edges = composers.make_first(inner, exception_type=GraphTestException)
        result = run.to_callable(edges)(y=_ROOT_VALUE, z=10)

        self.assertEqual(result.result, f"node1(node4(root, z=10))")

    def test_curry_with_state(self):

        edges = composers.make_first(
            curried_stateful_node, exception_type=GraphTestException
        )

        cg = run.to_callable(edges)
        result = cg(arg1=_ROOT_VALUE, arg2="arg2")
        result = cg(arg1=_ROOT_VALUE, arg2="arg2", state=result.state)
        result = cg(arg1=_ROOT_VALUE, arg2="arg2", state=result.state)

        self.assertEqual(
            result.result, f"curried_stateful_node(arg1=root, arg2=arg2, state=3)"
        )

    def test_state_is_serializable(self):
        edges = (
            graph.make_edge(source=node1, destination=reducer_node, key="arg1"),
            graph.make_edge(source=reducer_node, destination=node2, key="arg1"),
        )

        cg = run.to_callable(edges)
        result = cg(arg1=_ROOT_VALUE)
        result = cg(arg1=_ROOT_VALUE, state=result.state)
        result = cg(arg1=_ROOT_VALUE, state=result.state)

        json.dumps(result.state)

    def test_compose_compose(self):

        inner_graph = composers.make_compose(
            curried_node, node1, node3, node2, key="arg1"
        )
        self.assertEqual(len(inner_graph), 3)

        edges = composers.make_compose(inner_graph, node4, key="arg2")

        result = run.to_callable(edges)(y="y", z="z", arg1="arg1")

        self.assertEqual(len(edges), 5)
        self.assertEqual(
            result.result,
            "curried_node(arg1=node1(node3(arg1=node2(arg1), arg2=node4(y, z=z))), arg2=node4(y, z=z))",
        )

    def test_compose_after_first(self):
        edges = composers.make_compose(
            composers.make_first(
                unactionable_node, node1, node2, exception_type=GraphTestException
            ),
            node3,
            key="arg1",
        )
        result = run.to_callable(edges)(arg1="arg1", arg2="arg2")
        self.assertEqual(result.result, "node1(node3(arg1=arg1, arg2=arg2))")

    def test_first_after_compose(self):
        inner_edges = composers.make_compose(node1, node2)

        cg = run.to_callable(
            composers.make_first(
                unactionable_node, inner_edges, node1, exception_type=GraphTestException
            )
        )

        result = cg(arg1="arg1")
        self.assertEqual(result.result, "node1(node2(arg1))")

    def test_first_first(self):
        inner_first = composers.make_first(
            unactionable_node, node1, node2, exception_type=GraphTestException
        )

        cg = run.to_callable(
            composers.make_first(
                unactionable_node, inner_first, node1, exception_type=GraphTestException
            )
        )

        result = cg(arg1=_ROOT_VALUE)

        self.assertEqual(result.result, f"node1({_ROOT_VALUE})")

    def test_compose_with_node_already_in_graph(self):
        inner_edges1 = composers.make_and((node2, node1), merge_fn=merger)
        inner_edges2 = composers.make_compose(node3, node1, key="arg1")
        edges = composers.make_compose(inner_edges1, inner_edges2, key="arg1")

        result = run.to_callable(edges)(
            arg1=_ROOT_VALUE, arg2="arg2", side_effects="side_effects"
        )

        self.assertEqual(
            result.result,
            f"[node2(node3(arg1=node1(root), arg2=arg2)),node1(root)], side_effects=side_effects",
        )

    def test_first_with_subgraph_that_raises(self):
        inner = composers.make_compose(node2, unactionable_node)
        edges = composers.make_first(inner, node1, exception_type=GraphTestException)
        result = run.to_callable(edges)(arg1=_ROOT_VALUE)
        self.assertEqual(result.result, "node1(root)")

    def test_or_with_sink_that_raises(self):
        edges = composers.make_or(
            (unactionable_node, node1),
            merge_fn=merger_that_raises_when_empty,
            exception_type=GraphTestException,
        )
        result = run.to_callable(edges)(arg1=_ROOT_VALUE)

        self.assertEqual(result.result, "[node1(root)]")
