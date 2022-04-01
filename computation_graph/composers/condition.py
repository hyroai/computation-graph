from typing import Callable, Dict, Iterable

import gamla

from computation_graph import base_types, composers
from computation_graph.composers import lift, logic


def when(
    condition: base_types.GraphOrCallable,
    transformation: base_types.GraphOrCallable,
    source: base_types.GraphOrCallable,
) -> base_types.GraphType:
    """Transform the output of `source` using `transformation`, but only if `condition`'s output is truthy."""
    return if_then_else(
        condition, composers.compose_unary(transformation, source), source
    )


@gamla.curry
def require(
    condition: base_types.GraphOrCallable, result: base_types.GraphOrCallable
) -> base_types.GraphType:
    def check(x):
        if x:
            return None
        raise base_types.SkipComputationError

    return composers.make_and(
        (composers.compose_unary(check, condition), result),
        merge_fn=lambda args: args[1],
    )


case: Callable[[Dict[Callable, Callable]], base_types.GraphType] = gamla.compose_left(
    dict.items, gamla.map(gamla.star(require)), gamla.star(composers.make_first)
)


def require_all(
    conditions: Iterable[base_types.GraphType], graph: base_types.GraphType
) -> base_types.GraphType:
    return require(logic.all_true(conditions), graph)


def if_then_else(
    condition: base_types.GraphOrCallable,
    if_truthy: base_types.GraphOrCallable,
    if_falsy: base_types.GraphOrCallable,
) -> base_types.GraphType:
    def if_then_else(condition_value, true_value, false_value):
        if condition_value:
            return true_value
        return false_value

    return composers.compose_dict(
        if_then_else,
        {
            "condition_value": condition,
            "true_value": if_truthy,
            "false_value": if_falsy,
        },
    )


def require_or_default(default_value):
    return lambda condition, value: if_then_else(
        condition, value, lift.always(default_value)
    )
