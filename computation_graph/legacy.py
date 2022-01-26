import dataclasses
from typing import Any

import gamla

from computation_graph import base_types, composers


@dataclasses.dataclass(frozen=True)
class ComputationResult:
    result: Any
    state: Any


@gamla.curry
def handle_state(key, default, g):
    def retrieve_state(x):
        assert isinstance(x, ComputationResult), x
        return x.state

    return base_types.merge_graphs(
        composers.make_compose_future(g, retrieve_state, key, default),
        composers.compose_unary(retrieve_state, g),
        composers.compose_unary(gamla.attrgetter("result"), g),
    )
