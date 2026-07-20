from computation_graph.composers import lift


def test_always_returns_the_value():
    assert lift.always(5)() == 5
    obj = {"a": 1}
    assert lift.always(obj)() is obj


def test_always_name_keeps_scalars_readable():
    assert lift.always(5).__name__ == "always 5"
    assert lift.always(3.5).__name__ == "always 3.5"
    assert lift.always("cardiology").__name__ == "always cardiology"
    assert lift.always(None).__name__ == "always None"


def test_always_name_summarizes_containers_by_type_and_length():
    assert lift.always({"a": 1, "b": 2}).__name__ == "always dict[2]"
    assert lift.always([1, 2, 3]).__name__ == "always list[3]"
    assert lift.always((1, 2)).__name__ == "always tuple[2]"


def test_always_name_does_not_render_large_container_contents():
    # The point of the summary: naming a node must not stringify a huge value.
    name = lift.always({i: i for i in range(10_000)}).__name__
    assert name == "always dict[10000]"


def test_always_name_falls_back_to_type_for_other_objects():
    class Custom:
        pass

    assert lift.always(Custom()).__name__ == "always Custom"


def test_distinct_always_calls_produce_distinct_nodes_despite_equal_names():
    # Names are display-only; node identity is hash(func), and every always()
    # call is a fresh function, so equal names must not collapse two nodes.
    a, b = lift.always({"x": 1}), lift.always({"y": 2})
    assert a.__name__ == b.__name__ == "always dict[1]"
    assert a is not b
    assert hash(a) != hash(b)
