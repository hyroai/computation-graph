from computation_graph import composers, graph_runners


def test_unary_composition_with_graph_destination():
    result = graph_runners.nullary_infer_sink(
        composers.compose_unary(
            composers.compose_left(lambda: 1, lambda a, b: a - b, key="a"), lambda: 2
        )
    )

    assert result == -1


def test_ambiguity_does_not_blow_up():
    counter = 0

    def increment():
        nonlocal counter
        counter += 1

    def g():
        return "x"

    for _ in range(15):
        g = composers.make_first(
            composers.compose_left_unary(g, lambda x: increment() or x + "a"),
            composers.compose_left_unary(g, lambda x: increment() or x + "b"),
        )
    graph_runners.nullary_infer_sink(g)
    assert counter == 30
