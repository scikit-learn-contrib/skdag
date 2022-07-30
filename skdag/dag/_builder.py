from collections.abc import Mapping, Sequence

import networkx as nx
from skdag.dag._dag import DAG, DAGStep
from skdag.exceptions import DAGError

__all__ = ["DAGBuilder"]


class DAGBuilder:
    """
    Helper utility for creating a :class:`skdag.DAG`.

    ``DAGBuilder`` allows a graph to be defined incrementally by specifying one node
    (step) at a time. Graph edges are defined by providing optional dependency lists
    that reference each step by name. Note that steps must be defined before they are
    used as dependencies.

    See Also
    --------
    :class:`skdag.DAG` : The estimator DAG created by this utility.

    Examples
    --------

    >>> from skdag import DAGBuilder
    >>> from sklearn.decomposition import PCA
    >>> from sklearn.impute import SimpleImputer
    >>> from sklearn.linear_model import LogisticRegression
    >>> dag = (
    ...     DAGBuilder()
    ...     .add_step("impute", SimpleImputer())
    ...     .add_step("vitals", "passthrough", deps={"impute": slice(0, 4)})
    ...     .add_step("blood", PCA(n_components=2, random_state=0), deps={"impute": slice(4, 10)})
    ...     .add_step("lr", LogisticRegression(random_state=0), deps=["blood", "vitals"])
    ...     .make_dag()
    ... )
    >>> print(dag.draw().strip())
    o    impute
    |\\
    o o    blood,vitals
    |/
    o    lr
    """

    def __init__(self):
        self.graph = nx.DiGraph()

    def add_step(self, name, est, deps=None):
        self._validate_name(name)
        if isinstance(deps, Sequence):
            deps = {dep: None for dep in deps}

        if deps is not None:
            self._validate_deps(deps)
        else:
            deps = {}

        step = DAGStep(name, est, deps=deps)
        self.graph.add_node(name, step=step)

        for dep in deps:
            # Since node is new, edges will never form a cycle.
            self.graph.add_edge(dep, name)

        self._validate_graph()

        return self

    def _validate_name(self, name):
        if not isinstance(name, str):
            raise KeyError(f"step names must be strings, got '{type(name)}'")

        if name in self.graph.nodes:
            raise KeyError(f"step with name '{name}' already exists")

    def _validate_deps(self, deps):
        if not isinstance(deps, Mapping) or not all(
            [isinstance(dep, str) for dep in deps]
        ):
            raise ValueError(
                "deps parameter must be a map of labels to indices, "
                f"got '{type(deps)}'."
            )

        missing = [dep for dep in deps if dep not in self.graph]
        if missing:
            raise ValueError(f"unresolvable dependencies: {', '.join(sorted(missing))}")

    def _validate_graph(self):
        if not nx.algorithms.dag.is_directed_acyclic_graph(self.graph):
            raise DAGError("Workflow is not a DAG.")

    def make_dag(self, **kwargs):
        self._validate_graph()
        # Give the DAG a read-only view of the graph.
        return DAG(graph=self.graph.copy(as_view=True), **kwargs)

    def _repr_html_(self):
        return self.make_dag()._repr_html_()
