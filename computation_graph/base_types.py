from __future__ import annotations

import dataclasses
import functools
import typing
from typing import Callable, Hashable, Optional, Tuple, Union

import gamla

Result = Hashable


def pretty_print_function_name(f: Callable) -> str:
    return f"{f.__code__.co_filename}:{f.__code__.co_firstlineno}:{f.__name__}"


_get_unary_input_typing = gamla.compose_left(
    typing.get_type_hints,
    gamla.when(gamla.inside("return"), gamla.remove_key("return")),
    dict.values,
    gamla.head,
)


def _mismatch_message(key, source: Callable, destination: Callable) -> str:
    return "\n".join(
        [
            "",
            f"source: {pretty_print_function_name(source)}",
            f"destination: {pretty_print_function_name(destination)}",
            f"key: {key}",
            str(typing.get_type_hints(source)["return"]),
            str(
                typing.get_type_hints(destination)[key]
                if key is not None
                else _get_unary_input_typing(destination)
            ),
        ]
    )


class ComputationGraphTypeError(Exception):
    pass


class SkipComputationError(Exception):
    pass


@dataclasses.dataclass(frozen=True)
class ComputationEdge:
    destination: ComputationNode
    priority: int
    key: str
    source: Optional[ComputationNode]
    args: Tuple[ComputationNode, ...]
    is_future: bool

    def __post_init__(self):
        assert bool(self.args) != bool(
            self.source
        ), f"Edge must have a source or args, not both: {self}"
        if (
            not self.args
            # TODO(uri): doesn't support `functools.partial`, suggested to drop support for it entirely.
            and not isinstance(self.source.func, functools.partial)
            and not isinstance(self.destination.func, functools.partial)
        ):
            if not gamla.composable(self.destination.func, self.source.func, self.key):
                raise ComputationGraphTypeError(
                    _mismatch_message(self.key, self.source.func, self.destination.func)
                )

    def __repr__(self):
        source_str = (
            "".join(map(str, self.args)) if self.source is None else str(self.source)
        )
        line = "...." if self.is_future else "----"
        return source_str + line + self.key + line + ">" + str(self.destination)


merge_graphs = gamla.compose_left(gamla.pack, gamla.concat, gamla.unique, tuple)


@dataclasses.dataclass(frozen=True)
class NodeSignature:
    is_args: bool
    kwargs: Tuple[str, ...]
    optional_kwargs: Tuple[str, ...]
    is_kwargs: bool


@dataclasses.dataclass(frozen=True)
class ComputationNode:
    name: str
    func: Callable
    signature: NodeSignature
    is_terminal: bool

    def __hash__(self):
        if self.is_terminal:
            return hash(self.name)
        return id(self.func)

    def __repr__(self):
        return self.name


node_implementation = gamla.attrgetter("func")
node_is_terminal = gamla.attrgetter("is_terminal")

edge_args = gamla.attrgetter("args")
edge_destination = gamla.attrgetter("destination")
edge_key = gamla.attrgetter("key")
edge_priority = gamla.attrgetter("priority")
edge_source = gamla.attrgetter("source")
edge_is_future = gamla.attrgetter("is_future")


def edge_sources(edge: ComputationEdge) -> Tuple[ComputationNode, ...]:
    return edge.args or (edge.source,)  # type: ignore


is_computation_graph = gamla.alljuxt(
    # Note that this cannot be set to `GraphType` (due to `is_instance` limitation).
    gamla.is_instance(tuple),
    gamla.allmap(gamla.is_instance(ComputationEdge)),
)


ambiguity_groups = gamla.compose(
    frozenset,
    gamla.filter(gamla.len_greater(1)),
    dict.values,
    gamla.groupby(gamla.juxt(edge_destination, edge_key, edge_priority)),
)


# We use a tuple to generate a unique id for each node based on the order of edges.
GraphType = Tuple[ComputationEdge, ...]
GraphOrCallable = Union[Callable, GraphType]
CallableOrNode = Union[Callable, ComputationNode]
CallableOrNodeOrGraph = Union[CallableOrNode, GraphType]
NodeOrGraph = Union[ComputationNode, GraphType]
EMPTY_GRAPH = ()
