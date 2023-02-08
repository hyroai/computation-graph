import setuptools

with open("README.md", "r") as fh:
    _LONG_DESCRIPTION = fh.read()


setuptools.setup(
    name="computation-graph",
    python_requires=">=3",
    version="42",
    long_description=_LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    packages=setuptools.find_namespace_packages(),
    install_requires=["gamla", "typeguard", "toposort", "immutables", "termcolor"],
    extras_require={"test": ["pygraphviz", "pytest>=5.4.0"]},
)
