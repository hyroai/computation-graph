from typing import Callable

import pytest

from computation_graph import graph_runners
from computation_graph.composers import condition


@pytest.mark.parametrize(
    "condition_fn,expected_result", [(lambda: True, "T"), (lambda: False, "F")]
)
def test_if_then_else(condition_fn: Callable[[], bool], expected_result: str):
    assert (
        graph_runners.nullary_infer_sink(
            condition.if_then_else(condition_fn, lambda: "T", lambda: "F")
        )
        == expected_result
    )
