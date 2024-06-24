import asyncio
import functools
import inspect
from typing import Dict

import gamla

from computation_graph import base_types, graph


def duplicate_function(func):
    if asyncio.iscoroutinefunction(func):

        @functools.wraps(func)
        async def inner(*args, **kwargs):
            return await func(*args, **kwargs)

        return inner

    @functools.wraps(func)
    def inner(*args, **kwargs):
        return func(*args, **kwargs)

    inner.__name__ = f"duplicate of {inner.__name__}"
    return inner


def _duplicate_computation_edge(get_duplicated_node):
    return gamla.compose_left(
        gamla.dataclass_transform("source", get_duplicated_node),
        gamla.dataclass_transform("destination", get_duplicated_node),
        gamla.dataclass_transform(
            "args", gamla.compose_left(gamla.map(get_duplicated_node), tuple)
        ),
    )


_duplicate_node = gamla.compose_left(
    base_types.node_implementation,
    gamla.when(
        gamla.compose_left(inspect.signature, gamla.attrgetter("parameters"), len),
        duplicate_function,
    ),
    graph.make_computation_node,
)

duplicate_node = _duplicate_node

_node_to_duplicated_node = gamla.compose_left(
    gamla.remove(base_types.node_is_terminal),
    gamla.map(gamla.pair_right(_duplicate_node)),
    dict,
    gamla.dict_to_getter_with_default(None),
)

duplicate_graph = gamla.compose_left(
    gamla.pair_with(gamla.compose_left(graph.get_all_nodes, _node_to_duplicated_node)),
    gamla.star(
        lambda get_duplicated_node, graph: gamla.pipe(
            graph,
            gamla.map(
                _duplicate_computation_edge(
                    gamla.when(get_duplicated_node, get_duplicated_node)
                )
            ),
            tuple,
        )
    ),
)

duplicate_function_or_graph = gamla.ternary(
    gamla.is_instance(tuple), duplicate_graph, duplicate_function
)


def safe_replace_sources(
    source_to_replacement_dict: Dict[
        base_types.CallableOrNode, base_types.CallableOrNodeOrGraph
    ],
    cg: base_types.GraphType,
) -> base_types.GraphType:
    source_to_replacement_dict = gamla.keymap(graph.make_computation_node)(
        source_to_replacement_dict
    )
    reachable_to_duplicate_map = gamla.pipe(
        gamla.graph_traverse_many(
            source_to_replacement_dict.keys(), graph.traverse_forward(cg)
        ),
        gamla.remove(gamla.contains(source_to_replacement_dict.keys())),
        _node_to_duplicated_node,
    )
    get_node_replacement = gamla.compose_left(
        gamla.lazyjuxt(
            gamla.dict_to_getter_with_default(None)(source_to_replacement_dict),
            reachable_to_duplicate_map,
            gamla.identity,
        ),
        gamla.find(gamla.identity),
    )

    def update_edge(e: base_types.ComputationEdge) -> base_types.GraphType:
        dest = base_types.edge_destination(e)
        g = graph.replace_node(dest, get_node_replacement(dest))((e,))
        for source in base_types.edge_sources(e):
            g = graph.replace_source(source, get_node_replacement(source), g)
        return g

    return base_types.merge_graphs(*(update_edge(e) for e in cg))
