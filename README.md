[![Build Status](https://travis-ci.com/hyroai/computation-graph.svg?branch=master)](https://travis-ci.com/hyroai/computation-graph)

A functional composition framework that supports:
1. State - functions which retain state for their next turn of action.
2. Ambiguity - non deterministic composition with priorities.
3. Injection of compositions into long pipelines (deep dependency injection).
4. Non cancerous asyncio support (planned).

`pip install computation-graph`

To deploy: `python setup.py sdist bdist_wheel; twine upload dist/*; rm -rf dist/;`
