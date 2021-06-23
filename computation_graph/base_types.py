from __future__ import annotations

import dataclasses
from typing import Any, Callable, Dict, Optional, Text, Tuple

import gamla


@dataclasses.dataclass(frozen=True)
class ComputationResult:
    result: Any
    state: Any


@dataclasses.dataclass(frozen=True)
class ComputationEdge:
    destination: ComputationNode
    priority: int
    # Either key+source, just source, just args.
    key: Optional[Text] = None
    source: Optional[ComputationNode] = None
    args: Tuple[ComputationNode, ...] = ()

    def __post_init__(self):
        assert bool(self.args) != bool(
            self.source
        ), "Edge must have a source or args, not both."

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
    is_args: bool = False
    kwargs: Tuple[Text, ...] = ()
    optional_kwargs: Tuple[Text, ...] = ()


edge_destination = gamla.attrgetter("destination")
edge_key = gamla.attrgetter("key")
