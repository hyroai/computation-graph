import asyncio
import dataclasses
import functools
from typing import Dict

import gamla

from computation_graph import base_types, graph
from computation_graph.composers import debug


def duplicate_function(func):
    if asyncio.iscoroutinefunction(func):

        @functools.wraps(func)
        async def inner(*args, **kwargs):
            return await func(*args, **kwargs)

    else:

        @functools.wraps(func)
        def inner(*args, **kwargs):
            return func(*args, **kwargs)

    return debug.name_callable(inner, f"duplicate of {func.__name__}")


def _duplicate_computation_edge(get_duplicated_node):
    return gamla.compose_left(
        gamla.dataclass_transform("source", get_duplicated_node),
        gamla.dataclass_transform("destination", get_duplicated_node),
        gamla.dataclass_transform(
            "args", gamla.compose_left(gamla.map(get_duplicated_node), tuple)
        ),
    )


def _signature_is_empty(signature: base_types.NodeSignature) -> bool:
    return not (
        signature.kwargs
        or signature.optional_kwargs
        or signature.is_args
        or signature.is_kwargs
    )


def _duplicate_node(node: base_types.ComputationNode) -> base_types.ComputationNode:
    if _signature_is_empty(node.signature):
        return node
    inner = duplicate_function(node.func)
    return dataclasses.replace(node, name=inner.__name__, func=inner)


def _node_to_duplicated_node(nodes):
    return {node: _duplicate_node(node) for node in nodes if not node.is_terminal}.get


def duplicate_graph(original: base_types.GraphType) -> base_types.GraphType:

    _node_to_duplicated_node_index = gamla.pipe(
        original.edges, graph.get_all_nodes, _node_to_duplicated_node
    )
    get_duplicated_node = gamla.when(
        _node_to_duplicated_node_index, _node_to_duplicated_node_index
    )
    return base_types.GraphType(
        edges=gamla.pipe(
            original.edges,
            gamla.map(
                _duplicate_computation_edge(
                    gamla.when(get_duplicated_node, get_duplicated_node)
                )
            ),
            frozenset,
        ),
        sink=get_duplicated_node(original.sink),
    )


duplicate_function_or_graph = gamla.ternary(
    gamla.is_instance(base_types.GraphType), duplicate_graph, duplicate_function
)


def safe_replace_sources(
    source_to_replacement_dict: Dict[
        base_types.CallableOrNode, base_types.CallableOrNodeOrGraph
    ],
    cg: base_types.GraphType,
) -> base_types.GraphType:
    if not len(source_to_replacement_dict):
        return cg

    source_to_replacement_dict = gamla.pipe(
        source_to_replacement_dict,
        gamla.keymap(graph.make_computation_node),
        gamla.valmap(
            gamla.unless(
                gamla.is_instance(base_types.GraphType), graph.make_computation_node
            )
        ),
    )
    get_duplicate = gamla.pipe(
        gamla.graph_traverse_many(
            source_to_replacement_dict.keys(), graph.traverse_forward(cg.edges)
        ),
        gamla.remove(gamla.contains(source_to_replacement_dict.keys())),
        _node_to_duplicated_node,
    )

    def node_replacement(node):
        replacement = source_to_replacement_dict.get(node)
        if replacement is not None:
            return replacement
        duplicate = get_duplicate(node)
        return duplicate if duplicate is not None else node

    replacement_graphs_edges = {}

    def source_replacement(node):
        replacement = node_replacement(node)
        if isinstance(replacement, base_types.GraphType):
            replacement_graphs_edges[id(replacement)] = replacement.edges
            return replacement.sink
        return replacement

    new_edges = []
    for edge in cg.edges:
        destination = node_replacement(edge.destination)
        if edge.args:
            args = tuple(source_replacement(arg) for arg in edge.args)
            if destination is edge.destination and all(
                new is old for new, old in zip(args, edge.args)
            ):
                new_edges.append(edge)
            else:
                new_edges.append(
                    dataclasses.replace(edge, args=args, destination=destination)
                )
        else:
            source = source_replacement(edge.source)
            if source is edge.source and destination is edge.destination:
                new_edges.append(edge)
            else:
                new_edges.append(
                    dataclasses.replace(edge, source=source, destination=destination)
                )

    return base_types.GraphType(
        edges=base_types.merge_edges(
            frozenset(new_edges), *replacement_graphs_edges.values()
        ),
        sink=node_replacement(cg.sink),
    )
