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


def case(cases: Dict[Callable, Callable]) -> base_types.GraphType:
    return gamla.pipe(
        cases,
        dict.items,
        gamla.map(gamla.star(lambda predicate, fn: require(predicate, fn))),
        gamla.star(composers.make_first),
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
    def if_then_else(args):
        if args[0]:
            return args[1]
        return args[2]

    return composers.make_and((condition, if_truthy, if_falsy), merge_fn=if_then_else)


def require_or_default(default_value):
    return lambda condition, value: if_then_else(
        condition, value, lift.always(default_value)
    )


def lazy_if_else(
    condition: base_types.GraphOrCallable,
    do_if: base_types.GraphOrCallable,
    do_else: base_types.GraphOrCallable,
) -> base_types.GraphOrCallable:
    """Guarantees an exception is thrown on exactly one path, to avoid increasing ambiguity."""
    return composers.make_first(
        require(condition, do_if), require(logic.complement(condition), do_else)
    )
