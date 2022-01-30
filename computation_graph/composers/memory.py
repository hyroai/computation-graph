from typing import Any, Callable

import gamla

from computation_graph import base_types, composers, legacy


def accumulate(f: base_types.GraphOrCallable) -> base_types.GraphType:
    """Accumulate `f`'s results into a `tuple`."""

    @with_state("state", None)
    def accumulate(state, x):
        return *(state or ()), x

    return composers.make_compose(accumulate, f, key="x")


class _NoValueYet:
    pass


_NO_VALUE_YET = _NoValueYet()


def changed(f: base_types.GraphOrCallable) -> base_types.GraphType:
    @legacy.handle_state("memory", _NO_VALUE_YET)
    def check_changed(memory, value_to_watch):
        return legacy.ComputationResult(value_to_watch != memory, value_to_watch)

    return composers.make_compose(check_changed, f, key="value_to_watch")


def ever(bool_node):
    @with_state("state", None)
    def ever_inner(state, some_bool):
        return state or some_bool

    return composers.make_compose(ever_inner, bool_node, key="some_bool")


def reduce_with_past_result(
    reduce_with_past: Callable[[Any, Any], Any], f: base_types.GraphOrCallable
):
    @legacy.handle_state("state", None)
    def lag(state, current):
        return legacy.ComputationResult(state, current)

    return composers.compose_dict(
        reduce_with_past,
        {"previous": composers.make_compose(lag, f, key="current"), "current": f},
    )


@gamla.curry
def with_state(key: str, default, f: Callable) -> base_types.GraphType:
    return composers.make_compose_future(f, f, key, default)
