from __future__ import annotations

import dataclasses
import functools
import typing
from typing import Any, Callable, Dict, Optional, Text, Tuple

import gamla

from computation_graph import type_safety


@dataclasses.dataclass(frozen=True)
class ComputationResult:
    result: Any
    state: Any


def _mismatch_message(key, source: Callable, destination: Callable) -> str:
    return "\n".join(
        [
            f"Type mismatch for argument `{key}`. in composition {source.__name__} onto {destination.__name__} ",
            ":".join(
                map(str, [typing.get_type_hints(source)["return"], source.__code__])
            ),
            ":".join(
                map(
                    str, [typing.get_type_hints(destination)[key], destination.__code__]
                )
            ),
        ]
    )


@dataclasses.dataclass(frozen=True)
class ComputationEdge:
    destination: ComputationNode
    priority: int
    key: Optional[Text]
    source: Optional[ComputationNode]
    args: Tuple[ComputationNode, ...]

    def __post_init__(self):
        assert (
            not (self.args) or not self.key
        ), f"Edge with `args` cannot have a `key`: {self.args} {self.key}"
        assert bool(self.args) != bool(
            self.source
        ), f"Edge must have a source or args, not both: {self}"
        if (
            not self.args
            # TODO(uri): doesn't support `functools.partial`, suggested to drop support for it entirely.
            and not isinstance(self.source.func, functools.partial)
            and not isinstance(self.destination.func, functools.partial)
            # TODO(uri): Remove `ComputationResult` for more powerful type checks.
            and typing.get_type_hints(self.source.func).get("return")
            is not ComputationResult
        ):
            assert type_safety.can_compose(
                self.destination.func, self.source.func, self.key
            ), _mismatch_message(self.key, self.source.func, self.destination.func)

    def __repr__(self):
        source_str = (
            "".join(map(str, self.args)) if self.source is None else str(self.source)
        )
        return f"{source_str}----{self.key or ''}---->{self.destination}"


# We use a tuple to generate a unique id for each node based on the order of edges.
GraphType = Tuple[ComputationEdge, ...]


@dataclasses.dataclass(frozen=True)
class ComputationNode:
    name: Text
    func: Callable
    signature: NodeSignature
    is_stateful: bool
    is_terminal: bool

    def __hash__(self):
        if self.is_terminal:
            return hash(self.name)
        return id(self.func)

    def __repr__(self):
        return self.name


@dataclasses.dataclass(frozen=True)
class ComputationInput:
    kwargs: Dict[Text, Any]
    state: Any = None
    args: Tuple[Any, ...] = ()


@dataclasses.dataclass(frozen=True)
class NodeSignature:
    is_args: bool
    kwargs: Tuple[Text, ...]
    optional_kwargs: Tuple[Text, ...]
    is_kwargs: bool = False


edge_destination = gamla.attrgetter("destination")
edge_key = gamla.attrgetter("key")
