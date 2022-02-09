from computation_graph import graph_runners
from computation_graph.composers import memory


def test_changed():
    graph_runners.nullary_infer_sink_with_state_and_expectations(
        memory.changed(
            memory.with_state(
                "state", 0, lambda state: state + 1 if state < 2 else state
            )
        )
    )(True, True, False)
