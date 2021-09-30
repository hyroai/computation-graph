from typing import Collection, FrozenSet, Union

from computation_graph import type_safety


def test_no_annotations():
    def f(x):
        pass

    def g(x):
        pass

    assert type_safety.can_compose(f, g, None)


def test_simple_types():
    def f(x: int) -> str:
        pass

    def g(x: str) -> str:
        pass

    assert type_safety.can_compose(g, f, None)
    assert not type_safety.can_compose(f, g, None)


def test_compose_on_key():
    def f(x: int, y: str) -> str:
        pass

    def g(x: str) -> str:
        pass

    assert type_safety.can_compose(f, g, "y")
    assert not type_safety.can_compose(f, g, "x")


def test_compose_generics():
    def f(x: FrozenSet[int]) -> Collection[str]:
        pass

    def g(x: Collection[str]) -> str:
        pass

    assert type_safety.can_compose(g, f, None)
    assert not type_safety.can_compose(f, g, None)


def test_union():
    def f(x: Union[int, str]) -> Union[str, int]:
        pass

    def g(x: str) -> str:
        return x[0]

    assert type_safety.can_compose(f, g, None)
    assert not type_safety.can_compose(g, f, None)
