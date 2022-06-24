"""
Directed Acyclic Graphs (DAGs) may be used to construct complex workflows for
scikit-learn estimators. As the name suggests, data may only flow in one
direction and can't go back on itself to a previously run step.
"""
from tempfile import NamedTemporaryFile
from collections.abc import Sequence
import networkx as nx
import stackeddag.core as sd

from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.metaestimators import _BaseComposition


class DAGStep:
    """
    A single estimator step in a DAG.
    """

    def __init__(self, name, est):
        self.name = name
        self.estimator = est

        self._hashval = hash((type(self).__name__, self.name))

    def __repr__(self):
        return f"{type(self).__name__}({repr(self.name)}, {repr(self.estimator)})"

    def __hash__(self):
        return self._hashval


class DAGDisplay:
    _EMPTY = "[empty]"
    COLORS = {
        "light_red": "#ff7fa7",
        "light_blue": "#80c4fc",
        "light_green": "#75e6a3",
        "light_grey": "#e6e6e6",
        "white": "#ffffff",
        "black": "#000000",
    }

    def __init__(self, dag):
        self.dag = dag
        self.agraph = self.to_agraph()

    def _get_node_color(self, G, node):
        if len(list(G.predecessors(node))) == 0:
            return self.COLORS["light_red"]
        if len(list(G.successors(node))) > 0:
            return self.COLORS["light_blue"]
        return self.COLORS["light_green"]

    def _get_node_shape(self, G, node):
        if len(list(G.predecessors(node))) == 0:
            return "ellipse"
        if len(list(G.successors(node))) > 0:
            return "box"
        return "ellipse"

    def _is_empty(self):
        return len(self.dag) == 0

    def to_agraph(self):
        G = self.dag
        if self._is_empty():
            G = nx.DiGraph()
            G.add_node(
                self._EMPTY,
                color=self.COLORS["light_grey"],
                fontcolor="black",
                shape="box",
            )

        A = nx.nx_agraph.to_agraph(G)
        A.graph_attr["rankdir"] = "LR"
        for v in G.nodes:
            anode = A.get_node(v)
            anode.attr["fontname"] = G.nodes[v].get("fontname", "sans")
            anode.attr["shape"] = G.nodes[v].get("shape", self._get_node_shape(G, v))
            anode.attr["style"] = G.nodes[v].get("style", "filled")
            color = self._get_node_color(G, v)
            anode.attr["bgcolor"] = G.nodes[v].get("color", color)
            anode.attr["color"] = G.nodes[v].get("color", color)
            anode.attr["fontcolor"] = G.nodes[v].get("fontcolor", self.COLORS["white"])

        for u, v in A.edges():
            edge = A.get_edge(u, v)
            edge.attr["color"] = G.edges[u, v].get("color", self.COLORS["light_grey"])

        A.layout()
        return A

    def draw(self, format="svg", prog="dot"):
        A = self.agraph

        if format == "ascii":
            if self._is_empty():
                return self._EMPTY

            la = []
            for node in A.nodes():
                la.append((node, node))

            ed = []
            for src, dst in A.edges():
                ed.append((src, [dst]))

            return sd.edgesToText(sd.mkLabels(la), sd.mkEdges(ed))

        data = A.draw(format=format, prog="dot")

        if format == "svg":
            return data.decode(self.agraph.encoding)
        return data


class DAGBuilder:
    def __init__(self, n_jobs=None):
        self.graph = nx.DiGraph()
        self.n_jobs = n_jobs

    def add_step(self, name, est, deps=None):
        self._validate_name(name)
        step = DAGStep(name, est)
        self.graph.add_node(name, step=step)

        if deps is not None:
            self._validate_deps(deps)
            for dep in deps:
                # Since node is new, edges will never form a cycle.
                self.graph.add_edge(dep, name)

        self._validate_graph()

        return step

    def _validate_name(self, name):
        if not isinstance(name, str):
            raise KeyError(f"step names must be strings, got '{type(name)}'")

        if name in self.graph.nodes:
            raise KeyError(f"step with name '{name}' already exists")

    def _validate_deps(self, deps):
        if not isinstance(deps, Sequence) or not all(
            [isinstance(dep, (str, DAGStep)) for dep in deps]
        ):
            raise ValueError(
                "deps parameter must be a sequence of labels or DAGSteps, "
                f"got '{type(deps)}'."
            )

    def _validate_graph(self):
        if not nx.algorithms.dag.is_directed_acyclic_graph(self.graph):
            raise TypeError("Workflow is not a DAG.")

    def make_dag(self):
        self._validate_graph()
        return DAG(graph=self.graph, n_jobs=self.n_jobs)


class DAG(_BaseComposition):
    """
    A Directed Acyclic Graph (DAG) for scikit-learn estimator workflows.
    """

    # BaseEstimator interface
    _required_parameters = ["graph"]

    def __init__(self, graph, n_jobs=None):
        self.graph = graph
        self.n_jobs = n_jobs

    def get_params(self, deep=True):
        """Get parameters for this estimator.
        Returns the parameters given in the constructor as well as the
        estimators contained within the `steps` of the `Pipeline`.
        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        return self._get_params("graph", deep=deep)

    def set_params(self, **kwargs):
        """Set the parameters of this estimator.
        Valid parameter keys can be listed with ``get_params()``. Note that
        you can directly set the parameters of the estimators contained in
        `steps`.
        Parameters
        ----------
        **kwargs : dict
            Parameters of this estimator or parameters of estimators contained
            in `steps`. Parameters of the steps may be set using its name and
            the parameter name separated by a '__'.
        Returns
        -------
        self : object
            Pipeline class instance.
        """
        self._set_params("graph", **kwargs)
        return self

    def fit(self, X, y=None, **fit_params):
        """Fit the model.
        Fit all the transformers one after the other and transform the
        data. Finally, fit the transformed data using the final estimator.
        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of the
            pipeline.
        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps of
            the pipeline.
        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.
        Returns
        -------
        self : object
            Pipeline with fitted steps.
        """
        fit_params_steps = self._check_fit_params(**fit_params)
        Xt = self._fit(X, y, **fit_params_steps)
        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            if self._final_estimator != "passthrough":
                fit_params_last_step = fit_params_steps[self.steps[-1][0]]
                self._final_estimator.fit(Xt, y, **fit_params_last_step)

        return self

    def _validate_steps(self):
        self.steps_ = list(nx.topological_sort(self.graph))

        # validate names
        self._validate_names([step.name for step in self.steps_])

        # validate estimators
        transformers = []
        estimators = []
        for step in self.steps_:
            if self.graph.out_degree(step.name) > 0:
                transformers.append(step.estimator)
            else:
                estimators.append(step.estimator)

        for trf in transformers:
            if trf is None or trf == "passthrough":
                continue
            if not (
                hasattr(trf, "fit") or hasattr(trf, "fit_transform")
            ) or not hasattr(trf, "transform"):
                raise TypeError(
                    "All intermediate steps should be "
                    "transformers and implement fit and transform "
                    "or be the string 'passthrough' "
                    f"'{trf}' (type {type(trf)}) doesn't"
                )

        # We allow terminal estimators to be None as an identity transformation
        for est in estimators:
            if est is not None and est != "passthrough" and not hasattr(est, "fit"):
                raise TypeError(
                    "Last step of Pipeline should implement fit "
                    "or be the string 'passthrough'. "
                    f"'{est}' (type {type(est)}) doesn't"
                )

    def _repr_svg_(self):
        return DAGDisplay(self.steps).draw(format="svg")

    def _repr_png_(self):
        return DAGDisplay(self.steps).draw(format="png")

    def _repr_jpeg_(self):
        return DAGDisplay(self.steps).draw(format="jpg")

    def _repr_pretty_(self, p, cycle):
        if cycle:
            p.text(repr(self))
        else:
            p.text(str(self))

    def __str__(self):
        return DAGDisplay(self.steps).draw(format="ascii")
