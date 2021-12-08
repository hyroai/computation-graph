from typing import Iterable, Tuple

import gamla

from computation_graph import base_types, composers


def all_truthy(functions: Iterable[base_types.GraphOrCallable]) -> base_types.GraphType:
    def all_truthy(args: Tuple) -> bool:
        return all(args)

    return composers.make_and(functions, all_truthy)


def any_truthy(functions: Iterable[base_types.GraphOrCallable]) -> base_types.GraphType:
    def any_truthy(args: Tuple) -> bool:
        return any(args)

    return composers.make_and(functions, any_truthy)


all_true = gamla.compose_left(
    gamla.map(lambda graph: composers.compose_unary(gamla.equals(True), graph)),
    all_truthy,
)

all_false = gamla.compose_left(
    gamla.map(lambda graph: composers.compose_unary(gamla.equals(False), graph)),
    all_truthy,
)


def complement(function: base_types.GraphOrCallable) -> base_types.GraphType:
    def complement(value) -> bool:
        return not value

    return composers.compose_unary(complement, function)


def is_predicate(expected_value):
    def is_predicate(f):
        def is_expected_value(value):
            return value is expected_value

        is_expected_value.__name__ = f"is {expected_value!s:.30}"
        return composers.compose_unary(is_expected_value, f)

    return is_predicate


is_true = is_predicate(True)
is_false = is_predicate(False)
