from typing import Callable

import gamla

from computation_graph import base_types, composers


def always(x):
    def always():
        return x

    always.__name__ = f"always {x!s:.30}"
    return always


def function_to_graph(f: Callable) -> base_types.GraphType:
    # Note that the used identity function must be a new instance every time,
    # so can't be replace with something like `gamla.identity`.
    return composers.compose_unary(lambda x: x, f)


any_to_graph = gamla.case_dict(
    {
        base_types.is_computation_graph: gamla.identity,
        callable: function_to_graph,
        gamla.just(True): gamla.compose_left(always, function_to_graph),
    }
)
