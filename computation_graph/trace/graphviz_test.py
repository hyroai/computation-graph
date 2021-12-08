import os
import pathlib

import pygraphviz as pgv

from computation_graph import composers, graph, run
from computation_graph.trace import graphviz


def test_computation_trace(tmp_path: pathlib.Path):
    def node1(arg1):
        return f"node1({arg1})"

    def node2(arg1):
        return f"node2({arg1})"

    def unactionable_node(arg1):
        del arg1
        raise NotImplementedError

    filename = "visualize.dot"

    inner_edges = composers.make_compose(node1, node2)
    cg = run.to_callable_with_side_effect(
        graphviz.computation_trace(filename),
        graph.connect_default_terminal(
            composers.make_first(unactionable_node, inner_edges, node1)
        ),
        frozenset([NotImplementedError]),
    )

    cwd = os.getcwd()
    os.chdir(tmp_path)
    result = cg(arg1="arg1")
    assert (tmp_path / filename).exists()
    g = pgv.AGraph()
    g.read(str(tmp_path / filename))

    assert g.get_node(id(node1)).attr["result"] == "node1(node2(arg1))"
    assert g.get_node(id(node1)).attr["color"] == "red"
    assert g.get_node(id(node2)).attr["result"] == "node2(arg1)"
    assert g.get_node(id(node2)).attr["color"] == "red"

    assert not g.get_node(id(unactionable_node)).attr["color"]
    os.chdir(cwd)

    assert result.result[graph.DEFAULT_TERMINAL][0] == "node1(node2(arg1))"
