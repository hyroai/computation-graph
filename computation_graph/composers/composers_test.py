import time

from computation_graph import composers, graph, run


def test_ambiguity_does_not_blow_up():
    start_time = time.time()

    def g(x):
        return x

    for _ in range(15):
        g = composers.make_first(
            composers.compose_left_unary(g, lambda x: x + "a"),
            composers.compose_left_unary(g, lambda x: x + "b"),
        )
    run.to_callable(graph.connect_default_terminal(g), frozenset())(x="x")
    assert time.time() - start_time < 1
