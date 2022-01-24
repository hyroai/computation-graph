from typing import Callable, Union

import gamla

from computation_graph import base_types, composers, graph, run


def unary(g: base_types.GraphType, source: Callable, sink: Callable) -> Callable:
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
            prev = f({real_source: turn, **prev})
        return prev[graph.make_computation_node(sink)]

    return inner


def unary_with_state_infer_sink(g: base_types.GraphType, source: Callable) -> Callable:
    return unary_with_state(g, source, graph.infer_graph_sink(g))


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
            prev = f({**turn, **prev})
            assert prev[graph.make_computation_node(sink)] == expectation

    return inner


def variadic_infer_sink(g):
    f = run.to_callable_strict(g)

    def inner(*turns):
        prev = {}
        for turn in turns:
            prev = f({**turn, **prev})
        return prev[graph.infer_graph_sink(g)]

    return inner


def variadic_stateful_infer_sink(g):
    f = run.to_callable_strict(g)

    def inner(*turns):
        prev = {}
        for turn in turns:
            prev = f({**turn, **prev})
        return prev[graph.infer_graph_sink(g)]

    return inner


def nullary(g, sink):
    return gamla.compose(
        gamla.itemgetter(graph.make_computation_node(sink)),
        run.to_callable_strict(g),
        gamla.just({}),
    )()


def nullary_infer_sink(g):
    return nullary(g, graph.infer_graph_sink(g))


def nullary_infer_sink_with_state_and_expectations(g):
    f = run.to_callable_strict(g)

    def inner(*expectations):
        prev = {}
        for expectation in expectations:
            prev = f(prev)
            assert prev[graph.infer_graph_sink(g)] == expectation

    return inner
