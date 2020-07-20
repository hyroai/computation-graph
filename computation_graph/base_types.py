from __future__ import annotations

import dataclasses
from typing import Any, Callable, Dict, Optional, Text, Tuple


@dataclasses.dataclass(frozen=True)
class ComputationResult:
    result: Any
    state: Any


@dataclasses.dataclass(frozen=True)
class ComputationEdge:
    destination: ComputationNode
    priority: int
    # Either key+source, just source, just args.
    key: Optional[Text]
    source: Optional[ComputationNode]
    args: Tuple[ComputationNode, ...]
    is_future: bool

    def __post_init__(self):
        assert bool(self.args) != bool(
            self.source,
        ), "Edge must have a source or args, not both."

        assert not self.is_future or not bool(
            self.args,
        ), "future edges do not support args source"

        assert (
            not self.is_future or self.priority == 0
        ), "future edges cannot have priority"

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
    func: Optional[Callable]
    signature: NodeSignature

    def __hash__(self):
        return id(self.func)

    def __repr__(self):
        return self.name


@dataclasses.dataclass(frozen=True)
class ComputationInput:
    kwargs: Dict[Text, Any]
    args: Tuple[Any, ...] = ()


@dataclasses.dataclass(frozen=True)
class NodeSignature:
    is_args: bool = False
    kwargs: Tuple[Text, ...] = ()
    optional_kwargs: Tuple[Text, ...] = ()


class ExhaustedAllComputationPaths(Exception):
    pass
