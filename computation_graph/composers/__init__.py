from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple

import gamla

from computation_graph import base_types, graph, signature
from computation_graph.base_types import GraphType
from computation_graph.graph import make_computation_node


class _ComputationError:
    pass


_to_computation_node_if_callable = gamla.unless(
    gamla.is_instance(base_types.GraphType), graph.make_computation_node
)


def _get_edges_from_node_or_graph(
    node_or_graph: base_types.NodeOrGraph,
) -> base_types.GraphType:
    if isinstance(node_or_graph, base_types.ComputationNode):
        return base_types.EMPTY_GRAPH
    return node_or_graph


@gamla.curry
def make_optional(
    func: base_types.CallableOrNodeOrGraph, default_value: Any
) -> base_types.GraphType:
    return make_first(func, lambda: default_value)


def make_and(
    funcs: Iterable[base_types.CallableOrNodeOrGraph],
    merge_fn: base_types.CallableOrNodeOrGraph,
) -> base_types.GraphType:
    """Aggregate funcs' output into `merge_fn`.
    * merge_fn should have only 1 argument named `args`.
    * All funcs must not raise an exception in order for merge_fn to run.
    >>>make_and(composers.make_and([gamla.just(1), gamla.just(2), gamla.just(3)], lambda args: sum(args)))
    (justjustjust----*args---->args_to_tuple, args_to_tuple----args----><lambda>)
    Will return 6.
    """

    def args_to_tuple(*args):
        return args

    merge_node = graph.make_computation_node(args_to_tuple)
    # `merge_fn` may be a callable/node or a whole graph; `make_compose` handles
    # both, connecting `merge_node`'s output into `merge_fn` under key "args".
    merge_fn_graph = make_compose(merge_fn, merge_node, key="args")

    return gamla.sync.pipe(
        funcs,
        gamla.sync.map(_to_computation_node_if_callable),
        tuple,
        gamla.sync.juxtcat(
            gamla.sync.filter(gamla.is_instance(base_types.GraphType)),
            gamla.sync.compose_left(
                gamla.sync.map(
                    lambda node_or_graph: (
                        node_or_graph.sink
                        if isinstance(node_or_graph, base_types.GraphType)
                        else node_or_graph
                    )
                ),
                tuple,
                lambda nodes: base_types.GraphType(
                    frozenset(
                        [
                            base_types.ComputationEdge(
                                is_future=False,
                                priority=0,
                                source=None,
                                args=nodes,
                                destination=merge_node,
                                key="*args",
                            )
                        ]
                    ),
                    merge_node,
                ),
                gamla.wrap_tuple,
            ),
        ),
        tuple,
        lambda graphs: graph.merge_graphs(
            *graphs, merge_fn_graph, sink_node_or_graph=merge_fn_graph
        ),
    )


def make_or(
    nodes_or_graphs: Sequence[base_types.CallableOrNodeOrGraph],
    merge_fn: base_types.CallableOrNodeOrGraph,
) -> base_types.GraphType:
    """Aggregate funcs' output into `merge_fn`.
    * merge_fn should have only 1 argument named `args`.
    * merge_fn will run anyway, with the outputs of the funcs that didn't raise an exception.
    """

    def filter_computation_errors(*args):
        return gamla.pipe(
            args, gamla.remove(gamla.is_instance(_ComputationError)), tuple
        )

    filter_node = graph.make_computation_node(filter_computation_errors)
    # `merge_fn` may be a callable/node or a whole graph; `make_compose` handles
    # both, connecting `filter_node`'s output into `merge_fn` under key "args".
    merge_fn_graph = make_compose(merge_fn, filter_node, key="args")

    return gamla.sync.pipe(
        nodes_or_graphs,
        gamla.sync.map(make_optional(default_value=_ComputationError())),
        tuple,
        gamla.juxtcat(
            gamla.identity,
            gamla.sync.compose_left(
                gamla.sync.map(gamla.attrgetter("sink")),
                tuple,
                lambda sinks: GraphType(
                    frozenset(
                        [
                            base_types.ComputationEdge(
                                is_future=False,
                                priority=0,
                                source=None,
                                args=sinks,
                                destination=filter_node,
                                key="*args",
                            )
                        ]
                    ),
                    filter_node,
                ),
                gamla.wrap_tuple,
            ),
        ),
        tuple,
        lambda graphs: graph.merge_graphs(
            *graphs, merge_fn_graph, sink_node_or_graph=merge_fn_graph
        ),
    )


def make_first(*graphs: base_types.CallableOrNodeOrGraph) -> base_types.GraphType:
    """Returns a graph that when run, returns the first value that doesn't raise an exception.
    >>> def raise_some_exception():
    ...     raise Exception
    ... make_first(raise_some_exception, gamla.just(1))
    (raise_exception----constituent_of_first---->first_sink, just----constituent_of_first---->first_sink)
    Will return 1.
    """
    graph_or_nodes = tuple(map(_to_computation_node_if_callable, graphs))

    def first_sink(constituent_of_first):
        return constituent_of_first

    sink = graph.make_computation_node(first_sink)

    return graph.merge_graphs(
        *gamla.pipe(
            graph_or_nodes,
            gamla.map(_get_edges_from_node_or_graph),
            gamla.filter(gamla.is_instance(GraphType)),
        ),
        gamla.pipe(
            graph_or_nodes,
            gamla.map(
                lambda graph_or_node: (
                    graph_or_node.sink
                    if isinstance(graph_or_node, GraphType)
                    else graph_or_node
                )
            ),
            enumerate,
            gamla.map(
                gamla.star(
                    lambda i, g: base_types.ComputationEdge(
                        source=g,
                        destination=sink,
                        key="constituent_of_first",
                        args=(),
                        priority=i,
                        is_future=False,
                    )
                )
            ),
            frozenset,
            lambda edges: GraphType(edges, sink),
        ),
        sink_node_or_graph=sink,
    )


def last(*args) -> base_types.GraphType:
    return make_first(*reversed(args))


def _determine_sink(
    source: base_types.NodeOrGraph, destination: base_types.NodeOrGraph, is_future: bool
) -> base_types.ComputationNode:
    if is_future:
        if isinstance(source, base_types.GraphType):
            return source.sink
        return (
            destination.sink
            if isinstance(destination, base_types.GraphType)
            else destination
        )

    if isinstance(destination, base_types.GraphType):
        return destination.sink

    if destination.is_terminal:
        return source.sink if isinstance(source, base_types.GraphType) else source
    return destination


def _determine_composition_sink(
    graphs: Tuple[base_types.NodeOrGraph, ...]
) -> base_types.NodeOrGraph:
    for graph_or_node in reversed(graphs):
        if isinstance(graph_or_node, base_types.GraphType):
            return graph_or_node.sink

        if graph_or_node.is_terminal:
            continue

    raise AssertionError("Could not compose, failed to. find a sink")


@gamla.curry
def _try_connect(
    source: base_types.ComputationNode,
    key: Optional[str],
    priority: int,
    is_future: bool,
    destination: base_types.ComputationNode,
    unbound_destination_signature: base_types.NodeSignature,
) -> base_types.ComputationEdge:
    if key is None and signature.is_unary(unbound_destination_signature):
        key = gamla.head(signature.parameters(unbound_destination_signature))
    assert key is not None and key in signature.parameters(
        unbound_destination_signature
    ), f"Expecting a graph with key `{key}` but got `{destination}`"
    return base_types.ComputationEdge(
        source=source,
        destination=destination,
        key=key,
        args=(),
        priority=priority,
        is_future=is_future,
    )


@gamla.curry
def _infer_composition_edges(
    priority: int,
    key: Optional[str],
    is_future: bool,
    source: base_types.NodeOrGraph,
    destination: base_types.NodeOrGraph,
) -> base_types.GraphType:

    try_connect = _try_connect(
        source.sink if isinstance(source, GraphType) else source,
        key,
        priority,
        is_future,
    )

    if isinstance(destination, base_types.ComputationNode):
        sink = _determine_sink(source, destination, is_future)
        return graph.merge_graphs(
            base_types.GraphType(
                edges=frozenset([try_connect(destination, destination.signature)]),
                sink=sink,
            ),
            _get_edges_from_node_or_graph(source),
            sink_node_or_graph=sink,
        )
    unbound_signature = graph.unbound_signature(
        graph.get_incoming_edges_for_node(destination.edges)
    )
    return graph.merge_graphs(
        gamla.sync.pipe(
            destination,
            gamla.attrgetter("edges"),
            gamla.sync.mapcat(graph.get_edge_nodes),
            gamla.unique,
            gamla.sync.filter(
                gamla.sync.compose_left(
                    unbound_signature,
                    lambda sig: key in sig.kwargs
                    or (key is None and signature.is_unary(sig)),
                )
            ),
            # Do not add edges to nodes from source that are already present in destination (cycle).
            gamla.sync.filter(
                lambda node: isinstance(source, base_types.ComputationNode)
                or node
                not in graph.get_all_nodes(
                    source.edges if isinstance(source, GraphType) else source
                )
            ),
            gamla.sync.map(
                lambda destination_node: try_connect(
                    destination=destination_node,
                    unbound_destination_signature=unbound_signature(destination_node),
                )
            ),
            frozenset,
            gamla.sync.check(
                gamla.identity,
                AssertionError(
                    f"Cannot compose, destination signature does not contain key '{key}'"
                ),
            ),
            # Sink is a placeholder here; `merge_graphs` below sets the real sink
            # via `sink_node_or_graph`.
            lambda edges: GraphType(edges, None),  # type: ignore[arg-type]
        ),
        _get_edges_from_node_or_graph(source),
        destination,
        sink_node_or_graph=_determine_sink(source, destination, is_future),
    )


def _make_compose_inner(
    *funcs: base_types.CallableOrNodeOrGraph,
    key: Optional[str],
    is_future,
    priority: int,
) -> base_types.GraphType:
    assert (
        len(funcs) > 1
    ), f"Only {len(funcs)} function passed to compose, need at least 2, funcs={funcs}"
    return gamla.sync.pipe(
        funcs,
        reversed,
        gamla.sync.map(_to_computation_node_if_callable),
        gamla.sliding_window(2),
        gamla.sync.map(gamla.star(_infer_composition_edges(priority, key, is_future))),
        tuple,
        lambda graphs: graph.merge_graphs(
            *graphs, sink_node_or_graph=_determine_composition_sink(graphs)
        ),
    )


def make_compose(
    *funcs: base_types.CallableOrNodeOrGraph, key: Optional[str] = None
) -> base_types.GraphType:
    return _make_compose_inner(*funcs, key=key, is_future=False, priority=0)


def compose_unary(*funcs: base_types.CallableOrNodeOrGraph) -> base_types.GraphType:
    """Returns a graph of the funcs composed from right to left. All functions need to be unary (have 1 argument).
    >>> compose_unary(gamla.add(1), gamla.multiply(2), gamla.divide_by(2))
    (divide_by----y---->multiply, multiply----y---->add)
    """
    return _make_compose_inner(*funcs, key=None, is_future=False, priority=0)


def make_compose_future(
    destination: base_types.CallableOrNodeOrGraph,
    source: base_types.CallableOrNodeOrGraph,
    key: Optional[str],
    default: base_types.Result,
) -> base_types.GraphType:
    def when_memory_unavailable():
        return default

    if not isinstance(destination, base_types.GraphType):
        destination = make_computation_node(destination)

    if not isinstance(source, base_types.GraphType):
        source = make_computation_node(source)

    if isinstance(source, base_types.GraphType):
        sink_node_or_graph = source.sink
    elif isinstance(destination, base_types.GraphType):
        sink_node_or_graph = destination.sink
    else:
        sink_node_or_graph = destination
    return graph.merge_graphs(
        _make_compose_inner(destination, source, key=key, is_future=True, priority=0),
        _make_compose_inner(
            destination, when_memory_unavailable, key=key, is_future=False, priority=1
        ),
        sink_node_or_graph=sink_node_or_graph,
    )


def compose_unary_future(
    destination: base_types.CallableOrNodeOrGraph,
    source: base_types.CallableOrNodeOrGraph,
    default: base_types.Result,
) -> base_types.GraphType:
    return make_compose_future(destination, source, None, default)


def compose_source(
    destination: base_types.CallableOrNodeOrGraph,
    key: str,
    source: base_types.CallableOrNodeOrGraph,
) -> base_types.GraphType:
    return _make_compose_inner(destination, source, key=key, is_future=True, priority=0)


@gamla.curry
def compose_left_source(
    source: base_types.CallableOrNodeOrGraph,
    key: str,
    destination: base_types.CallableOrNodeOrGraph,
):
    return compose_source(destination, key, source)


def compose_source_unary(
    destination: base_types.CallableOrNodeOrGraph,
    source: base_types.CallableOrNodeOrGraph,
) -> base_types.GraphType:
    return _make_compose_inner(
        destination, source, key=None, is_future=True, priority=0
    )


def compose_left(*args, key: Optional[str] = None) -> base_types.GraphType:
    """Compose a function onto another function on a certain key.
    >>>compose_left(gamla.just(1), gamla.between, key="low")
    (just----low---->between,)
    """
    return make_compose(*reversed(args), key=key)


def compose_left_future(
    source: base_types.GraphOrCallable,
    destination: base_types.GraphOrCallable,
    key: Optional[str],
    default: base_types.Result,
) -> base_types.GraphType:
    return make_compose_future(destination, source, key, default)


def compose_left_unary(*args) -> base_types.GraphType:
    """Returns a graph of the funcs composed from left to right. All functions need to be unary (have 1 argument).
    >>> compose_left_unary(gamla.add(1), gamla.multiply(2), gamla.divide_by(2))
    (add----y---->multiply, multiply----y---->divide_by)
    """
    return compose_unary(*reversed(args))


@gamla.curry
def compose_dict(
    f: base_types.GraphOrCallable, d: Dict[str, base_types.CallableOrNodeOrGraph]
) -> base_types.GraphType:
    """Compose the functions in d.values() onto f, where d.keys() specify the arguments
    >>> compose_dict(gamla.between, {"low": gamla.just(0), "high": gamla.just(10)})
    (just----low---->between, just----high---->between)
    """

    destination = f if base_types.is_computation_graph(f) else make_computation_node(f)

    return gamla.pipe(
        d,
        dict.items,
        gamla.sync.map(gamla.star(lambda key, fn: make_compose(destination, fn, key=key))),
        tuple,
        lambda graphs: graph.merge_graphs(
            *graphs, sink_node_or_graph=destination
        ),
    ) or compose_left_unary(f, lambda x: x)


@gamla.curry
def compose_left_dict(d: Dict, f: base_types.GraphOrCallable) -> base_types.GraphType:
    return compose_dict(f, d)


def make_raise_exception(exception):
    def inner():
        raise exception

    return inner


def side_effect(f):
    def side_effect(g):
        return compose_left_unary(g, gamla.side_effect(f))

    return side_effect


@gamla.curry
def compose_many_to_one(
    aggregation: Callable, graphs: Iterable[base_types.GraphOrCallable]
):
    return make_and(graphs, aggregation)


@gamla.curry
def aggregation(
    aggregation: Callable[[Iterable], Any], graphs: Iterable[base_types.GraphOrCallable]
) -> base_types.GraphType:
    """Same as `compose_many_to_one`, but takes care to duplicate `aggregation`, and allows it to have any arg name."""
    return make_and(
        graphs,
        # It is important that `aggregation` is duplicated here.
        # If it weren't for the `compose_left` we would need to do it explicitly.
        gamla.compose_left(lambda args: args, aggregation),
    )
