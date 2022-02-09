import gamla

from computation_graph import base_types


def is_edge_participating(in_trace_nodes):
    def is_edge_participating(edge):
        return in_trace_nodes(edge.destination) and gamla.anymap(in_trace_nodes)(
            [edge.source, *edge.args]
        )

    return is_edge_participating


def get_edge_label(edge: base_types.ComputationEdge) -> str:
    if edge.key in (None, "args"):
        return ""
    if edge.key == "first_input":
        return str(edge.priority)
    return edge.key or ""
