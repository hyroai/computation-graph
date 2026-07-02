import dataclasses
from typing import Any

import gamla

from computation_graph import base_types, composers, graph


@dataclasses.dataclass(frozen=True)
class ComputationResult:
    result: Any
    state: Any


@gamla.curry
def handle_state(
    key: str, default: Any, g: base_types.GraphOrCallable
) -> base_types.GraphType:
    @graph.make_terminal("retrieve_state")
    def retrieve_state(x):
        if x == ():
            raise base_types.SkipComputationError
        x = x[0]
        assert isinstance(x, ComputationResult), x
        return x.state

    def sink(r: ComputationResult):
        return r.result

    return graph.merge_graphs(
        composers.make_compose_future(g, retrieve_state, key, default),
        composers.compose_unary(retrieve_state, g),
        composers.compose_unary(sink, g),
        sink_node_or_graph=graph.make_computation_node(sink),
    )
