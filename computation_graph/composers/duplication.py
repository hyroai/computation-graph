import asyncio
import functools
import inspect

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

_node_to_duplicated_node = gamla.compose_left(
    graph.get_all_nodes,
    gamla.remove(base_types.node_is_terminal),
    gamla.map(gamla.pair_right(_duplicate_node)),
    dict,
    gamla.dict_to_getter_with_default(None),
)

duplicate_graph = gamla.compose_left(
    gamla.pair_with(_node_to_duplicated_node),
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

_traverse_forward = gamla.compose_left(
    gamla.mapcat(
        gamla.compose_left(
            gamla.juxt(base_types.edge_sources, base_types.edge_destination),
            gamla.explode(0),
        )
    ),
    gamla.groupby_many_reduce(
        gamla.compose_left(gamla.head, gamla.wrap_tuple),
        lambda ds, e: (*(ds if ds else ()), e[1]),
    ),
    gamla.dict_to_getter_with_default(()),
    gamla.before(graph.make_computation_node),
)


def safe_replace_node(o, r, g):
    for v in gamla.graph_traverse(o, _traverse_forward(g)):
        if v == o:
            g = graph.replace_node(o, r)(g)
        else:
            g = graph.replace_node(v, _duplicate_node(v))(g)

    return g
