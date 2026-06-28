from typing import Any, Callable

import gamla

from computation_graph import base_types, composers, legacy
from computation_graph.composers import coloring

# Every combinator here compares-or-accumulates across turns, so its inner node is an
# observer (`coloring.observer` marks it + pins its future-state machinery core). See
# coloring.observer: the activation builder does NOT exempt observers -- a skill-private
# one prunes with its skill (state survives via the reducer {**prev} latch). The mark
# drives diagnostics / an optional must-tick latch. A pure carry-forward LATCH is not an
# observer -- which is why `with_state` / `handle_state` themselves are not marked here.


def accumulate(f: base_types.GraphOrCallable) -> base_types.GraphType:
    """Accumulate `f`'s results into a `tuple`."""

    def accumulate(state, x):
        return *(state or ()), x

    return coloring.observer(
        composers.make_compose(with_state("state", None)(accumulate), f, key="x"),
        accumulate,
    )


class _NoValueYet:
    pass


_NO_VALUE_YET = _NoValueYet()


def changed(f: base_types.GraphOrCallable) -> base_types.GraphType:
    def check_changed(memory, value_to_watch):
        return legacy.ComputationResult(value_to_watch != memory, value_to_watch)

    return coloring.observer(
        composers.make_compose(
            legacy.handle_state("memory", _NO_VALUE_YET)(check_changed),
            f,
            key="value_to_watch",
        ),
        check_changed,
    )


def ever(bool_node):
    def ever_inner(state, some_bool):
        return state or some_bool

    return coloring.observer(
        composers.make_compose(
            with_state("state", None)(ever_inner), bool_node, key="some_bool"
        ),
        ever_inner,
    )


def reduce_with_past_result(
    reduce_with_past: Callable[[Any, Any], Any], f: base_types.GraphOrCallable
):
    def lag(state, current):
        return legacy.ComputationResult(state, current)

    return coloring.observer(
        composers.compose_dict(
            reduce_with_past,
            {
                "previous": composers.make_compose(
                    legacy.handle_state("state", None)(lag), f, key="current"
                ),
                "current": f,
            },
        ),
        lag,
    )


@gamla.curry
def with_state(key: str, default, f: Callable) -> base_types.GraphType:
    return composers.make_compose_future(f, f, key, default)
