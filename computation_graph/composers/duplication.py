import functools

import gamla

from computation_graph import graph


def duplicate_function(func):
    @functools.wraps(func)
    def inner(*args, **kwargs):
        return func(*args, **kwargs)

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
    gamla.attrgetter("func"), duplicate_function, graph.make_computation_node
)

_node_to_duplicated_node = gamla.compose_left(
    graph.get_all_nodes,
    gamla.map(gamla.pair_right(_duplicate_node)),
    dict,
    gamla.dict_to_getter_with_default(None),
)

duplicate_graph = gamla.compose_left(
    gamla.pair_with(_node_to_duplicated_node),
    gamla.star(
        lambda get_duplicated_node, graph: gamla.pipe(
            graph, gamla.map(_duplicate_computation_edge(get_duplicated_node)), tuple
        )
    ),
)

duplicate_function_or_graph = gamla.ternary(
    gamla.is_instance(tuple), duplicate_graph, duplicate_function
)
