from typing import FrozenSet

from computation_graph import base_types, graph


def infer_sink(
    edges: FrozenSet[base_types.ComputationEdge],
) -> base_types.ComputationNode:
    future_edges = tuple(e for e in edges if e.is_future)
    non_future_edges = tuple(e for e in edges if not e.is_future)

    if not non_future_edges:
        all_future_sources = frozenset(
            source for e in future_edges for source in base_types.edge_sources(e)
        )
        future_destinations = frozenset(e.destination for e in future_edges)
        leaf_destinations = future_destinations - all_future_sources
        if not leaf_destinations:
            # Self-referential cycle: every destination is also a source (e.g. a -future-> a).
            leaf_destinations = future_destinations & all_future_sources
        assert len(leaf_destinations) == 1, leaf_destinations
        return next(iter(leaf_destinations))

    all_nodes = graph.get_all_nodes(non_future_edges)
    # Only edges to non-terminal destinations count: a node that only sources
    # edges into terminals is still a leaf (terminals are side-effect sinks).
    all_sources = frozenset(
        source
        for e in non_future_edges
        if not e.destination.is_terminal
        for source in base_types.edge_sources(e)
    )
    leaves = all_nodes - all_sources

    terminals = frozenset(n for n in leaves if n.is_terminal)
    sources_of_future = frozenset(
        source for e in future_edges for source in base_types.edge_sources(e)
    )

    candidates = leaves - terminals - sources_of_future
    if len(candidates) == 1:
        return next(iter(candidates))

    non_terminal_leaves = leaves - terminals
    if len(non_terminal_leaves) == 1:
        return next(iter(non_terminal_leaves))

    assert False, f"Could not infer unique sink; candidates: {non_terminal_leaves}"


# get_edge_nodes = opt_gamla.ternary( # This is a duplication with graph.py
#     base_types.edge_args,
#     lambda edge: edge.args + (edge.destination,),
#     lambda edge: (edge.source, edge.destination),
# )
#
# def get_all_nodes_with_filter( # This is a duplication with graph.py
#     filter_nodes: Optional[Callable[[base_types.ComputationNode], bool]],
# ):
#     f = opt_gamla.mapcat(get_edge_nodes)
#
#     if filter_nodes is not None:
#         f = opt_gamla.compose_left(f, frozenset, opt_gamla.filter(filter_nodes))
#
#     # return functools.cache(
#     #     gamla.timeit_with_label(
#     #         "get_all_nodes_with_filter", opt_gamla.compose_left(f, frozenset)
#     #     )
#     # )
#
#     return functools.cache(
#             opt_gamla.compose_left(f, frozenset))
#
#
#
#
# def _sink_excluding_terminals(edges: base_types.GraphType) -> base_types.ComputationNode:
#     leaves_not_terminal = frozenset(
#         node for node in _get_leaves(edges) if not node.is_terminal
#     )
#     assert (
#         len(leaves_not_terminal) == 1
#     ), f"Expected exactly one non-terminal sink, got {leaves_not_terminal}"
#     return gamla.head(leaves_not_terminal)
#
# _remove_future_edges = opt_gamla.compose(
#     tuple, opt_gamla.remove(base_types.edge_is_future)
# )
#
# def _get_leaves(edges: base_types.GraphType) -> FrozenSet[base_types.ComputationNode]:
#     all_nodes = graph.get_all_nodes(edges)
#     all_destinations = frozenset(
#         gamla.concat((edge.source, *edge.args) for edge in edges)
#     )
#     return all_nodes - all_destinations
#
# # @gamla.timeit_with_label("infer_sink")
# def infer_sink_old(graph_or_node:frozenset[ComputationEdge]) -> base_types.ComputationNode:
#     if isinstance(graph_or_node, base_types.ComputationNode):
#         return graph_or_node
#     # logging.info(f"infer_sink {len(graph_or_node)}")
#     graph_without_future_edges = _remove_future_edges(graph_or_node)
#     if graph_without_future_edges:
#         try:
#             return _sink_excluding_terminals(graph_without_future_edges)
#         except AssertionError:
#             # If we reached here we can try again without sources of future edges.
#             sources_of_future_edges = gamla.sync.pipe(
#                 graph_or_node,
#                 gamla.sync.filter(base_types.edge_is_future),
#                 gamla.sync.map(base_types.edge_source),
#                 frozenset,
#             )
#             result = (
#                 _get_leaves(graph_without_future_edges) - sources_of_future_edges
#             )
#             assert len(result) == 1, result
#             return gamla.head(result)
#
#     assert len(_destinations(graph_or_node)) == 1, graph_or_node
#     return base_types.edge_destination(gamla.head(graph_or_node))
#
#
# _destinations = gamla.compose(set, gamla.map(base_types.edge_destination))
