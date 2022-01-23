import dataclasses
from typing import Any

import gamla

from computation_graph import base_types, composers, graph


@dataclasses.dataclass(frozen=True)
class LegacyComputationResult:
    result: Any
    state: Any


def handle_state(g):
    def retrieve_state(x):
        assert isinstance(x, LegacyComputationResult), x
        return x.state

    @graph.make_terminal(f"state sink for {g}")
    def state_terminal(x):
        return x

    return base_types.merge_graphs(
        composers.make_compose_future(g, retrieve_state, "s"),
        composers.compose_unary(retrieve_state, g),
        composers.compose_unary(gamla.attrgetter("result"), g),
        composers.compose_unary(state_terminal, retrieve_state),
    )
