from typing import Any, Callable

import gamla

from computation_graph import base_types, composers, graph


def accumulate(f: base_types.GraphOrCallable) -> base_types.GraphType:
    """Accumulate `f`'s results into a `tuple`."""

    def accumulate(state, x):
        return *(state or ()), x

    return composers.make_compose(accumulate, f, key="x")


def changed(f: base_types.GraphOrCallable) -> base_types.GraphType:
    def check_changed(state, x):
        return base_types.ComputationResult(x != state, x)

    return composers.make_compose(check_changed, f, key="x")


def ever(bool_node):
    def ever_inner(state, some_bool):
        return state or some_bool

    return composers.make_compose(ever_inner, bool_node, key="some_bool")


def reduce_with_past_result(
    reduce_with_past: Callable[[Any, Any], Any], f: base_types.GraphOrCallable
):
    def lag(state, current):
        return base_types.ComputationResult(state, current)

    return composers.compose_dict(
        reduce_with_past,
        {"previous": composers.make_compose(lag, f, key="current"), "current": f},
    )


@gamla.curry
def with_state(key: str, f: Callable) -> base_types.GraphType:
    return (graph.make_future_edge(source=f, destination=f, key=key),)
