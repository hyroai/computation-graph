from typing import Callable

import pytest

from computation_graph import graph, run
from computation_graph.composers import condition


@pytest.mark.parametrize(
    "condition_fn,expected_result", [(lambda: True, "T"), (lambda: False, "F")]
)
def test_if_then_else(condition_fn: Callable[[], bool], expected_result: str):
    result = run.to_callable(
        graph.connect_default_terminal(
            condition.if_then_else(condition_fn, lambda: "T", lambda: "F")
        ),
        frozenset(),
    )()

    assert result.result[graph.DEFAULT_TERMINAL][0] == expected_result
