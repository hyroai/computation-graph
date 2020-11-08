[![Build Status](https://travis-ci.com/hyroai/computation-graph.svg?branch=master)](https://travis-ci.com/hyroai/computation-graph)

A functional composition framework that supports:

1. State - functions which retain state for their next turn of action.
2. Ambiguity - non deterministic composition with priorities.
3. Injection of compositions into long pipelines (deep dependency injection).
4. Non cancerous asyncio support.

`pip install computation-graph`

To deploy: `python setup.py sdist bdist_wheel; twine upload dist/*; rm -rf dist/;`

### Debugging

We need graphviz to visualize computation graphs:

```bash
sudo apt update && apt install graphviz
pip install pygraphviz
```

Debugging is possible by setting
`config.DEBUG_SAVE_COMPUTATION_TRACE = True` or environment variable CG_DEBUG_SAVE_COMPUTATION_TRACE to true/t/1.
This will save a file, on each graph execution, named `computation.dot` to current working directory.

You can use this file in a graph viewer like [gephi](https://gephi.org/).
Nodes colored red are part of the 'winning' computation path.
Each of these nodes has the attributes 'result' and 'state'.
'result' is the output of the node, and 'state' is the _new_ state of the node.

In gephi you can filter for the nodes participating in calculation of final result by filtering on result != null.
