from computation_graph import base_types
from computation_graph.composers import memory


def _nullary_with_state_and_expectations():
    raise NotImplementedError


def test_changed():
    _nullary_with_state_and_expectations(
        memory.changed(
            lambda state: base_types.ComputationResult(
                state, 0 if state is None else state + 1
            )
        )
    )(False, True)
