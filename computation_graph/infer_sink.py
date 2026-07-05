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
