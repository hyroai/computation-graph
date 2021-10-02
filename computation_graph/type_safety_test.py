from typing import (
    Any,
    Collection,
    FrozenSet,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import pytest

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


@pytest.mark.parametrize(
    "subtype,supertype",
    [
        [None, Optional[str]],
        [None, None],
        [Optional[str], Optional[str]],
        [Optional[str], Union[None, str]],
        [str, Optional[str]],
        [FrozenSet[str], FrozenSet[str]],
        [str, Any],
        [Tuple[str, ...], Tuple[str, ...]],
        [Set[str], Collection[str]],
        [List, Sequence],
        [Union[int, str], Union[int, str]],
        [str, Union[int, str]],
        [Union[List, Set], Collection],
    ],
)
def test_is_subtype(subtype, supertype):
    assert type_safety.is_subtype(subtype, supertype)


@pytest.mark.parametrize(
    "subtype,supertype",
    [
        [Optional[str], None],
        [Optional[str], Optional[int]],
        [int, Optional[str]],
        [FrozenSet[int], FrozenSet[str]],
        [str, FrozenSet[str]],
        [Collection, FrozenSet],
        [Tuple[str, ...], Tuple[int, ...]],
        [Union[int, str], int],
        [Any, str],
        [List, Union[int, str]],
        [Union[int, str, List], Union[int, str]],
    ],
)
def test_is_not_subtype(subtype, supertype):
    assert not type_safety.is_subtype(subtype, supertype)
