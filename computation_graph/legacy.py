import dataclasses
from typing import Any

import gamla

from computation_graph import base_types, composers


@dataclasses.dataclass(frozen=True)
class LegacyComputationResult:
    result: Any
    state: Any


def handle_state(g):
    def retrieve_state(x):
        assert isinstance(x, LegacyComputationResult), x
        return x.state

    return base_types.merge_graphs(
        composers.make_compose_future(g, retrieve_state, "state"),
        composers.compose_unary(retrieve_state, g),
        composers.compose_unary(gamla.attrgetter("result"), g),
    )
