from __future__ import annotations

import dataclasses
import functools
import os
import typing
from typing import Callable, FrozenSet, Hashable, Optional, Tuple, Union

import gamla

COMPUTATION_GRAPH_DEBUG_ENV_KEY = "COMPUTATION_GRAPH_DEBUG"

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
    # Edges are hashed on every frozenset construction/lookup; the fields are
    # immutable so the hash is computed once. Excluded from __eq__/__repr__.
    computed_hash: int = dataclasses.field(init=False, repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(
            self,
            "computed_hash",
            hash(
                (
                    self.destination,
                    self.priority,
                    self.key,
                    self.source,
                    self.args,
                    self.is_future,
                )
            ),
        )
        assert bool(self.args) != bool(
            self.source
        ), f"Edge must have a source or args, not both: {self}"
        assert bool(self.args) or isinstance(
            self.source, ComputationNode
        ), f"source must be a ComputationNode {self.source}"
        assert isinstance(
            self.destination, ComputationNode
        ), f"destination must be a ComputationNode {self.destination}"
        assert all(
            isinstance(x, ComputationNode) for x in self.args
        ), f"all args must be ComputationNodes {self.args}"
        if (
            not self.args
            and not isinstance(self.source.func, functools.partial)
            and not isinstance(self.destination.func, functools.partial)
        ):
            if not gamla.composable(self.destination.func, self.source.func, self.key):
                raise ComputationGraphTypeError(
                    _mismatch_message(self.key, self.source.func, self.destination.func)
                )

    def __hash__(self):
        return self.computed_hash

    def __repr__(self):
        source_str = (
            f"[{''.join(map(str, self.args))}]"
            if self.source is None
            else str(self.source)
        )
        line = "...." if self.is_future else "----"
        return source_str + line + self.key + line + ">" + str(self.destination)


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
    computed_hash: int = dataclasses.field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "computed_hash", hash(self.func))

    def __hash__(self):
        return self.computed_hash

    def __repr__(self):
        return f"<CompuationNode {self.name.replace('<', '').replace('>', '')}({','.join(self.signature.kwargs)}) >"


@dataclasses.dataclass(frozen=True)
class GraphType:
    edges: FrozenSet[ComputationEdge]
    sink: ComputationNode

    def __post_init__(self):
        assert isinstance(self.edges, frozenset), "edges must be a frozenset"

    # The properties below are derived from `edges` (immutable), computed on
    # first access and cached on the instance. Graph transformations that know
    # how a derived value changes may pre-seed it via `seed_nodes` instead of
    # letting the next reader recompute it from scratch.

    @functools.cached_property
    def nodes(self) -> FrozenSet[ComputationNode]:
        result = set()
        for edge in self.edges:
            if edge.source is not None:
                result.add(edge.source)
            else:
                result.update(edge.args)
            result.add(edge.destination)
        return frozenset(result)

    @functools.cached_property
    def node_to_incoming_edges(
        self,
    ) -> typing.Mapping[ComputationNode, FrozenSet[ComputationEdge]]:
        grouped: dict = {}
        for edge in self.edges:
            grouped.setdefault(edge.destination, []).append(edge)
        return {node: frozenset(group) for node, group in grouped.items()}

    @functools.cached_property
    def node_to_forward_neighbors(
        self,
    ) -> typing.Mapping[ComputationNode, FrozenSet[ComputationNode]]:
        grouped: dict = {}
        for edge in self.edges:
            for source in edge.args or (edge.source,):
                grouped.setdefault(source, set()).add(edge.destination)
        return {node: frozenset(group) for node, group in grouped.items()}

    def seed_nodes(self, nodes: FrozenSet[ComputationNode]) -> "GraphType":
        """Pre-populate the `nodes` cache when the caller already knows it."""
        self.__dict__["nodes"] = nodes
        return self


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
    gamla.is_instance(GraphType),
    gamla.compose_left(gamla.attrgetter("edges"), gamla.len_greater(0)),
    gamla.compose_left(
        gamla.attrgetter("edges"), gamla.allmap(gamla.is_instance(ComputationEdge))
    ),
)


_ambiguity_groups_on_edges = gamla.compose(
    frozenset,
    gamla.filter(gamla.len_greater(1)),
    dict.values,
    gamla.keyfilter(gamla.compose_left(gamla.head, gamla.complement(node_is_terminal))),
    gamla.groupby(gamla.juxt(edge_destination, edge_key, edge_priority)),
)

assert_no_unwanted_ambiguity_on_edges = gamla.side_effect(
    gamla.compose_left(
        _ambiguity_groups_on_edges,
        gamla.assert_that_with_message(
            gamla.wrap_str(
                "There are multiple edges with the same destination, key and priority in the computation graph!: {}"
                f"\n To get a more relevant stacktrace please set the environment variable with %env {COMPUTATION_GRAPH_DEBUG_ENV_KEY}=1 and rebuild."
            ),
            gamla.len_equals(0),
        ),
    )
)


def is_debug_mode() -> bool:
    return os.getenv(COMPUTATION_GRAPH_DEBUG_ENV_KEY) is not None


@gamla.side_effect
def assert_no_unwanted_ambiguity_when_debug_set(graph: GraphType):
    if is_debug_mode():
        assert_no_unwanted_ambiguity_on_edges(graph.edges)


# TODO(nitzo): Is this used ? Check if we can re-enable the assertion when in debug / move to merge.
def merge_edges(*edges: FrozenSet[ComputationEdge]) -> FrozenSet[ComputationEdge]:
    result: FrozenSet[ComputationEdge] = frozenset()
    return result.union(*edges)


GraphOrCallable = Union[Callable, GraphType]
CallableOrNode = Union[Callable, ComputationNode]
CallableOrNodeOrGraph = Union[CallableOrNode, GraphType]
NodeOrGraph = Union[ComputationNode, GraphType]
EMPTY_GRAPH: GraphType = GraphType(edges=frozenset(), sink=None)  # type: ignore[arg-type]
