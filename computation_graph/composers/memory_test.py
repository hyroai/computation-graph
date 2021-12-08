from computation_graph import base_types, graph, run
from computation_graph.composers import memory


def test_changed():
    cg = run.to_callable(
        graph.connect_default_terminal(
            memory.changed(
                lambda state: base_types.ComputationResult(
                    state, 0 if state is None else state + 1
                )
            )
        ),
        frozenset(),
    )
    result = cg()
    assert not result.result[graph.DEFAULT_TERMINAL][0]
    result = cg(state=result.state)
    assert result.result[graph.DEFAULT_TERMINAL][0]
