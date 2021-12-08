from typing import Callable

import gamla

from computation_graph import base_types, composers


def always(x):
    def always():
        return x

    always.__name__ = f"always {x!s:.30}"
    return always


def function_to_graph(f: Callable) -> base_types.GraphType:
    return composers.compose_unary(gamla.identity, f)


any_to_graph = gamla.case_dict(
    {
        base_types.is_computation_graph: gamla.identity,
        callable: function_to_graph,
        gamla.just(True): gamla.compose_left(always, function_to_graph),
    }
)
