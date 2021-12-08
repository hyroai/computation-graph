[![Build Status](https://travis-ci.com/hyroai/computation-graph.svg?branch=master)](https://travis-ci.com/hyroai/computation-graph)

A functional composition framework that supports:

1. State - functions which retain state for their next turn of action.
2. Ambiguity - non deterministic composition with priorities.
3. Injection of compositions into long pipelines (deep dependency injection).
4. Non cancerous `asyncio` support.

`pip install computation-graph`

To deploy: `python setup.py sdist bdist_wheel; twine upload dist/*; rm -rf dist/;`

### Type checking

The runner will type check all outputs for nodes with return type annotations. In case of a wrong typing, it will log the node at fault.

### Debugging

#### Computation trace

Available computation trace visualizers:

1. `graphviz.computation_trace`
1. `mermaid.computation_trace`
1. `ascii.computation_trace`

To use, replace `to_callable` with `run.to_callable_with_side_effect` with your selected style as the first argument.

#### Graphviz debugger

This debugger will save a file on each graph execution to current working directory.

You can use this file in a graph viewer like [gephi](https://gephi.org/).
Nodes colored red are part of the 'winning' computation path.
Each of these nodes has the attributes 'result' and 'state'.
'result' is the output of the node, and 'state' is the _new_ state of the node.

In gephi you can filter for the nodes participating in calculation of final result by filtering on result != null.
