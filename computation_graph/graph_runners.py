from typing import Callable, Union

import gamla

from computation_graph import base_types, composers, graph, run


def _infer_graph_sink(edges: base_types.GraphType) -> base_types.ComputationNode:
    assert edges, "Empty graphs have no sink."
    leaves = graph.get_leaves(edges)
    assert len(leaves) == 1, f"Cannot determine sink for {edges}, got: {tuple(leaves)}."
    return gamla.head(leaves)


def unary(g: base_types.GraphType, source: Callable, sink: Callable) -> Callable:
    return gamla.compose(
        gamla.itemgetter(graph.make_computation_node(sink)), unary_bare(g, source)
    )


def unary_bare(g, source):
    real_source = graph.make_source()
    return gamla.compose(
        gamla.star(
            run.to_callable_strict(
                base_types.merge_graphs(
                    g, composers.compose_left_future(real_source, source, None, None)
                )
            )
        ),
        gamla.pair_with(gamla.just({})),
        gamla.wrap_dict(real_source),
    )


def unary_with_state(
    g: base_types.GraphType,
    source: Callable,
    sink: Union[Callable, base_types.ComputationNode],
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
            prev = f(prev, {real_source: turn})
        return prev[graph.make_computation_node(sink)]

    return inner


def unary_with_state_infer_sink(g: base_types.GraphType, source: Callable) -> Callable:
    return unary_with_state(g, source, _infer_graph_sink(g))


def unary_with_state_and_expectations(
    g: base_types.GraphType, source: Callable, sink: Callable
) -> Callable:
    real_source = graph.make_source()
    return gamla.compose(
        variadic_with_state_and_expectations(
            base_types.merge_graphs(
                g, composers.compose_left_future(real_source, source, None, None)
            ),
            sink,
        ),
        tuple,
        gamla.map(lambda t: ({real_source: t[0]}, t[1])),
    )


def variadic_with_state_and_expectations(g, sink):
    f = run.to_callable_strict(g)

    def inner(turns):
        prev = {}
        for turn, expectation in turns:
            prev = f(prev, turn)
            assert (
                prev[graph.make_computation_node(sink)] == expectation
            ), f"actual={prev[graph.make_computation_node(sink)]}\n expected: {expectation}"

    return inner


def variadic_bare(g):
    f = run.to_callable_strict(g)

    def inner(*turns):
        prev = {}
        for turn in turns:
            prev = f(prev, turn)
        return prev

    return inner


def variadic_infer_sink(g):
    return gamla.compose_left(variadic_bare(g), gamla.itemgetter(_infer_graph_sink(g)))


def variadic_stateful_infer_sink(g):
    f = run.to_callable_strict(g)
    sink = _infer_graph_sink(g)

    def inner(*turns):
        prev = {}
        for turn in turns:
            prev = f(prev, turn)
        return prev[sink]

    return inner


def nullary(g, sink):
    return gamla.compose(
        gamla.itemgetter(graph.make_computation_node(sink)), run.to_callable_strict(g)
    )({}, {})


def nullary_infer_sink(g):
    return nullary(g, _infer_graph_sink(g))


def nullary_infer_sink_with_state_and_expectations(g):
    f = run.to_callable_strict(g)
    sink = _infer_graph_sink(g)

    def inner(*expectations):
        prev = {}
        for expectation in expectations:
            prev = f(prev, {})
            assert prev[sink] == expectation

    return inner
