import os
import pathlib

import pygraphviz as pgv

from computation_graph import base_types, composers, run
from computation_graph.trace import graphviz


def test_computation_trace(tmp_path: pathlib.Path):
    def node1(x):
        return f"node1({x})"

    def node2():
        return "node2"

    def raises():
        raise base_types.SkipComputationError

    filename = "visualize.dot"
    f = run.to_callable_with_side_effect(
        graphviz.computation_trace(filename),
        composers.make_first(raises, composers.compose_unary(node1, node2)),
        frozenset(),
    )
    cwd = os.getcwd()
    os.chdir(tmp_path)
    f({}, {})
    assert (tmp_path / filename).exists()
    g = pgv.AGraph()
    g.read(str(tmp_path / filename))
    assert g.get_node(id(node1)).attr["result"] == "node1(node2)"
    assert g.get_node(id(node1)).attr["color"] == "red"
    assert g.get_node(id(node2)).attr["result"] == "node2"
    assert g.get_node(id(node2)).attr["color"] == "red"
    assert not g.get_node(id(raises)).attr["color"]
    os.chdir(cwd)
