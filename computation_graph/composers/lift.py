from typing import Callable

import gamla

from computation_graph import base_types, composers


def _short_name(x) -> str:
    if x is None or isinstance(x, (int, float, str)):
        return f"{x!s:.30}"
    if isinstance(x, (dict, list, tuple, set, frozenset)):
        return f"{type(x).__name__}[{len(x)}]"
    return type(x).__name__


def always(x):
    def always():
        return x

    always.__name__ = f"always {_short_name(x)}"
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
