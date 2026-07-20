import pytest

from computation_graph import composers, graph
from computation_graph.infer_sink import infer_sink

# Case 1: only regular ComputationNodes


def test_case1_chain_of_regular_nodes():
    def a():
        return "a"

    def b(x):
        return x

    def c(x):
        return x

    g = composers.compose_left_unary(a, composers.compose_left_unary(b, c))
    assert infer_sink(g.edges) == graph.make_computation_node(c)


def test_case1_single_edge():
    def a():
        return "a"

    def b(x):
        return x

    g = composers.compose_left_unary(a, b)
    assert infer_sink(g.edges) == graph.make_computation_node(b)


# Case 2: mix of computation nodes and terminals


def test_case2_terminal_sibling_of_sink():
    def a():
        return "a"

    def b(x):
        return x

    def c(x):
        return x

    t = graph.make_terminal("t", lambda x: x)
    g = graph.merge_graphs(
        composers.compose_left_unary(a, composers.compose_left_unary(b, c)),
        composers.compose_unary(t, b),
        sink_node_or_graph=graph.make_computation_node(c),
    )
    assert infer_sink(g.edges) == graph.make_computation_node(c)


def test_case2_multiple_terminals_along_path():
    def a():
        return "a"

    def b(x):
        return x

    def c(x):
        return x

    t1 = graph.make_terminal("t1", lambda x: x)
    t2 = graph.make_terminal("t2", lambda x: x)
    g = graph.merge_graphs(
        composers.compose_left_unary(a, composers.compose_left_unary(b, c)),
        composers.compose_unary(t1, b),
        composers.compose_unary(t2, c),
        sink_node_or_graph=graph.make_computation_node(c),
    )
    assert infer_sink(g.edges) == graph.make_computation_node(c)


# Case 3: only terminals are leaves


def test_case3_single_terminal_child():
    def a():
        return "a"

    def b(x):
        return x

    t = graph.make_terminal("t", lambda x: x)
    g = graph.merge_graphs(
        composers.compose_left_unary(a, b),
        composers.compose_unary(t, b),
        sink_node_or_graph=graph.make_computation_node(b),
    )
    assert infer_sink(g.edges) == graph.make_computation_node(b)


def test_case3_multiple_terminal_children_single_parent():
    def a():
        return "a"

    t1 = graph.make_terminal("t1", lambda x: x)
    t2 = graph.make_terminal("t2", lambda x: x)
    g = graph.merge_graphs(
        composers.compose_unary(t1, a),
        composers.compose_unary(t2, a),
        sink_node_or_graph=graph.make_computation_node(a),
    )
    assert infer_sink(g.edges) == graph.make_computation_node(a)


# Case 4: mix of computation nodes and future edges


def test_case4_future_edge_from_intermediate_node():
    def a():
        return "a"

    def b(x):
        return x

    def c(x):
        return x

    def mem(x):
        return x

    g = graph.merge_graphs(
        composers.compose_left_unary(a, composers.compose_left_unary(b, c)),
        composers.compose_left_source(b, "x", mem),
        sink_node_or_graph=graph.make_computation_node(c),
    )
    assert infer_sink(g.edges) == graph.make_computation_node(c)


def test_case4_sink_is_source_of_future_edge():
    def a():
        return "a"

    def b(x):
        return x

    def mem(x):
        return x

    g = graph.merge_graphs(
        composers.compose_left_unary(a, b),
        composers.compose_left_source(b, "x", mem),
        sink_node_or_graph=graph.make_computation_node(b),
    )
    assert infer_sink(g.edges) == graph.make_computation_node(b)


# Case 5: only future edges are leaves


def test_case5_only_future_edge_outgoing_from_sink():
    def a():
        return "a"

    def b(x):
        return x

    def mem(x):
        return x

    g = graph.merge_graphs(
        composers.compose_left_unary(a, b),
        composers.compose_left_source(b, "x", mem),
        sink_node_or_graph=graph.make_computation_node(b),
    )
    assert infer_sink(g.edges) == graph.make_computation_node(b)


def test_case5_only_future_edges_single_destination():
    s = graph.make_source()

    def dest(c, b):
        return b

    dest_node = graph.make_computation_node(dest)
    g = graph.merge_graphs(
        composers.compose_left_source(s, "c", dest),
        composers.compose_left_source(s, "b", dest),
        sink_node_or_graph=dest_node,
    )
    assert infer_sink(g.edges) == dest_node


# Case 6: mix of computation nodes, future edges and terminals


def test_case6_full_mix():
    def a():
        return "a"

    def b(x):
        return x

    def c(x):
        return x

    def mem(x):
        return x

    t = graph.make_terminal("t", lambda x: x)
    g = graph.merge_graphs(
        composers.compose_left_unary(a, composers.compose_left_unary(b, c)),
        composers.compose_unary(t, b),
        composers.compose_left_source(c, "x", mem),
        sink_node_or_graph=graph.make_computation_node(c),
    )
    assert infer_sink(g.edges) == graph.make_computation_node(c)


def test_case6_sink_is_terminal_parent_and_future_source():
    def a():
        return "a"

    def b(x):
        return x

    def mem(x):
        return x

    t = graph.make_terminal("t", lambda x: x)
    g = graph.merge_graphs(
        composers.compose_left_unary(a, b),
        composers.compose_unary(t, b),
        composers.compose_left_source(b, "x", mem),
        sink_node_or_graph=graph.make_computation_node(b),
    )
    assert infer_sink(g.edges) == graph.make_computation_node(b)


# make_first


def test_make_first_two_nodes():
    g = composers.make_first(lambda: 1, lambda: 2)
    assert infer_sink(g.edges) == g.sink


def test_make_first_with_composed_graph():
    def a():
        return "a"

    def b(x):
        return x

    g = composers.make_first(composers.compose_left_unary(a, b), lambda: "fallback")
    assert infer_sink(g.edges) == g.sink


# make_and


def test_make_and_two_nodes():
    def merge(args):
        return args

    g = composers.make_and([lambda: 1, lambda: 2], merge)
    assert infer_sink(g.edges) == g.sink


def test_make_and_three_nodes():
    def merge(args):
        return args

    g = composers.make_and([lambda: 1, lambda: 2, lambda: 3], merge)
    assert infer_sink(g.edges) == g.sink


# make_or


def test_make_or_two_nodes():
    def merge(args):
        return args

    g = composers.make_or([lambda: 1, lambda: 2], merge)
    assert infer_sink(g.edges) == g.sink


def test_make_or_three_nodes():
    def merge(args):
        return args

    g = composers.make_or([lambda: 1, lambda: 2, lambda: 3], merge)
    assert infer_sink(g.edges) == g.sink


# Invalid graphs


def test_invalid_two_disconnected_chains():
    def a():
        return "a"

    def b(x):
        return x

    def c():
        return "c"

    def d(x):
        return x

    edges = (
        composers.compose_left_unary(a, b).edges
        | composers.compose_left_unary(c, d).edges
    )
    with pytest.raises(AssertionError):
        infer_sink(edges)


def test_future_only_multiple_sources_single_destination():
    def a():
        return "a"

    def b():
        return "b"

    def c(x, y):
        return x

    edges = (
        composers.compose_left_source(a, "x", c).edges
        | composers.compose_left_source(b, "y", c).edges
    )
    assert infer_sink(edges) == graph.make_computation_node(c)


def test_invalid_terminals_multiple_parents():
    def a():
        return "a"

    def b():
        return "b"

    t = graph.make_terminal("t", lambda x: x)
    edges = composers.compose_unary(t, a).edges | composers.compose_unary(t, b).edges
    with pytest.raises(AssertionError):
        infer_sink(edges)


def test_future_composition_graph_source_sink_is_source_sink():
    """When make_compose_future receives a graph as source, the sink should be source.sink.
    dest must be inside source_graph (i.e., the feedback cycle pattern)."""

    def a():
        return 1

    def b():
        return 2

    def dest(x, y):
        return x + y

    def aggregator(args):
        return sum(args)

    source_graph = composers.make_and([a, b, dest], aggregator)
    g = composers.make_compose_future(dest, source_graph, "x", 0)
    assert g.sink == source_graph.sink


def test_future_composition_plain_source_sink_is_destination():
    """When make_compose_future receives a plain node as source, the sink should be destination."""

    def source():
        return 1

    def dest(x):
        return x + 1

    g = composers.make_compose_future(dest, source, "x", 0)
    assert g.sink == graph.make_computation_node(dest)


def test_complex_future_edge():
    def a():
        pass

    def b(x):
        pass

    def c(x):
        pass

    def d(x, y):
        pass

    t = graph.make_terminal("t", lambda x: x)

    g = graph.merge_graphs(
        composers.compose_left(a, c),
        composers.compose_left_future(d, b, "x", "bla"),
        composers.compose_left(a, t),
        composers.compose_left(c, d, key="x"),
        composers.compose_left(b, d, key="y"),
        sink_node_or_graph=graph.make_computation_node(d),
    )

    assert g.sink == infer_sink(g.edges)
