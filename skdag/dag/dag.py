"""
Directed Acyclic Graphs (DAGs) may be used to construct complex workflows for
scikit-learn estimators. As the name suggests, data may only flow in one
direction and can't go back on itself to a previously run step.
"""
from collections import UserDict
from collections.abc import Sequence

import networkx as nx
import numpy as np
from scipy import sparse
from skdag.dag.renderer import DAGRenderer
from sklearn.base import clone
from sklearn.utils import _print_elapsed_time
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.metaestimators import _BaseComposition, available_if
from sklearn.utils.validation import check_memory

__all__ = ["DAG", "DAGStep"]


def _is_passthrough(estimator):
    return estimator is None or estimator == "passthrough"


def _is_transformer(estimator):
    return (
        hasattr(estimator, "fit") or hasattr(estimator, "fit_transform")
    ) and hasattr(estimator, "transform")


def _is_predictor(estimator):
    return (hasattr(estimator, "fit") or hasattr(estimator, "fit_predict")) and hasattr(
        estimator, "predict"
    )


def _transform_one(transformer, X, y, weight, **fit_params):
    res = transformer.transform(X)
    # if we have a weight for this transformer, multiply output
    if weight is None:
        return res
    return res * weight


def _fit_transform_one(
    transformer,
    X,
    y,
    weight,
    message_clsname="",
    message=None,
    allow_predictor=True,
    **fit_params,
):
    """
    Fits ``transformer`` to ``X`` and ``y``. The transformed result is returned
    with the fitted transformer. If ``weight`` is not ``None``, the result will
    be multiplied by ``weight``.
    """
    with _print_elapsed_time(message_clsname, message):
        failed = False
        if hasattr(transformer, "fit_transform"):
            res = transformer.fit_transform(X, y, **fit_params)
        elif hasattr(transformer, "transform"):
            res = transformer.fit(X, y, **fit_params).transform(X)
        elif allow_predictor:
            if hasattr(transformer, "fit_predict"):
                res = transformer.fit_predict(X, y, **fit_params)
            elif hasattr(transformer, "predict"):
                res = transformer.fit(X, y, **fit_params).predict(X)
            else:
                failed = True
                res = None

            if res is not None and res.ndim < 2:
                res = res.reshape(-1, 1)
        else:
            failed = True

        if failed:
            raise AttributeError(
                f"'{type(transformer).__name__}' object has no attribute 'transform'"
            )

    if weight is None:
        return res, transformer
    return res * weight, transformer


def _fit_one(transformer, X, y, weight, message_clsname="", message=None, **fit_params):
    """
    Fits ``transformer`` to ``X`` and ``y``.
    """
    with _print_elapsed_time(message_clsname, message):
        return transformer.fit(X, y, **fit_params)


def _leaf_estimators_have(attr, how="all"):
    """Check that leaves have `attr`.
    Used together with `avaliable_if` in `DAG`."""

    def check(self):
        # raise original `AttributeError` if `attr` does not exist
        failed = []
        for leaf in self._leaves:
            try:
                getattr(leaf.estimator, attr)
            except AttributeError:
                failed.append(leaf.estimator)

        if (how == "all" and failed) or (
            how == "any" and len(failed) != len(self._leaves)
        ):
            raise AttributeError(
                f"{', '.join([repr(type(est)) for est in failed])} object(s) has no attribute '{attr}'"
            )
        return True

    return check


def _stack(Xs, axis=0):
    if any(sparse.issparse(f) for f in Xs):
        if -2 <= axis < 2:
            axis = axis % 2
        else:
            raise NotImplementedError(
                f"Stacking is not supported for sparse inputs on axis > 1."
            )

        if axis == 0:
            Xs = sparse.vstack(Xs).tocsr()
        elif axis == 1:
            Xs = sparse.hstack(Xs).tocsr()
    else:
        if axis == 1:
            Xs = np.hstack(Xs)
        else:
            Xs = np.stack(Xs, axis=axis)

    return Xs


class DAGStep:
    """
    A single estimator step in a DAG.
    """

    def __init__(self, name, estimator, axis=1):
        self.name = name
        self.estimator = estimator
        self.axis = axis
        self.is_root = False
        self.is_leaf = False

        self._hashval = hash((type(self).__name__, self.name))

    def __repr__(self):
        return f"{type(self).__name__}({repr(self.name)}, {repr(self.estimator)})"

    def __hash__(self):
        return self._hashval


class DAG(_BaseComposition):
    """
    A Directed Acyclic Graph (DAG) for scikit-learn estimator workflows.
    """

    # BaseEstimator interface
    _required_parameters = ["graph"]

    @classmethod
    def from_pipeline(cls, steps, **kwargs):
        graph = nx.DiGraph()
        if hasattr(steps, "steps"):
            pipeline = steps
            steps = pipeline.steps
            if hasattr(pipeline, "get_params"):
                kwargs = {
                    **{
                        k: v
                        for k, v in pipeline.get_params().items()
                        if k in ("memory", "verbose")
                    },
                    **kwargs,
                }

        for i in range(len(steps)):
            name, estimator = steps[i]
            step = DAGStep(name, estimator)
            graph.add_node(name, step=step)
            if i > 0:
                dep = steps[i - 1][0]
                graph.add_edge(dep, name)

        return cls(graph=graph, **kwargs)

    def __init__(self, graph, *, memory=None, n_jobs=None, verbose=False):
        self.graph = graph
        self.memory = memory
        self.verbose = verbose
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

    def _log_message(self, step):
        if not self.verbose:
            return None

        return f"(step {step.name}: {step.index} of {len(self.graph_)}) Processing {step.name}"

    def _iter(self, with_leaves=True, filter_passthrough=True):
        """
        Generate stage lists from self.graph_.
        When filter_passthrough is True, 'passthrough' and None transformers
        are filtered out.
        """
        for stage in nx.topological_generations(self.graph_):
            stage = [self.graph_.nodes[step]["step"] for step in stage]
            if not with_leaves:
                stage = [step for step in stage if not step.is_leaf]

            if filter_passthrough:
                stage = [
                    step
                    for step in stage
                    if step.estimator is not None and step.estimator != "passthough"
                ]

            if len(stage) == 0:
                continue

            yield stage

    def __len__(self):
        """
        Returns the size of the DAG
        """
        return len(self.graph)

    def _fit(self, X, y=None, **fit_params_steps):
        # Setup the memory
        memory = check_memory(self.memory)

        fit_transform_one_cached = memory.cache(_fit_transform_one)

        Xin = self._resolve_inputs(X)
        Xs = {}
        for stage in self._iter(with_leaves=False, filter_passthrough=False):
            for step in stage:
                transformer = step.estimator
                if transformer is None or transformer == "passthrough":
                    with _print_elapsed_time("DAG", self._log_message(step)):
                        continue

                if hasattr(memory, "location") and memory.location is None:
                    # we do not clone when caching is disabled to
                    # preserve backward compatibility
                    cloned_transformer = transformer
                else:
                    cloned_transformer = clone(transformer)

                deps = list(self.graph_.predecessors(step.name))
                if deps:
                    X = _stack([Xs[dep] for dep in deps], axis=step.axis)
                else:
                    # For root nodes, the destination rather than the source is
                    # specified.
                    X = Xin[step.name]

                # Fit or load from cache the current transformer
                Xs[step.name], fitted_transformer = fit_transform_one_cached(
                    cloned_transformer,
                    X,
                    y,
                    None,
                    message_clsname=type(self).__name__,
                    message=self._log_message(step),
                    **fit_params_steps[step.name],
                )

                # If all of a dep's dependents are now complete, we can free up some
                # memory.
                for dep in deps:
                    dependents = self.graph_.successors(dep)
                    if all(d in Xs for d in dependents):
                        del Xs[dep]

                # Replace the transformer of the step with the fitted
                # transformer. This is necessary when loading the transformer
                # from the cache.
                step.estimator = fitted_transformer
        return Xs

    def _resolve_inputs(self, X):
        if isinstance(X, (dict, UserDict)):
            inputs = sorted(X.keys)
            if inputs != sorted(self._roots):
                raise ValueError(
                    "Input dicts must contain one key per entry node. "
                    f"Entry nodes are {self._roots}, got {inputs}."
                )
        else:
            if len(self._roots) != 1:
                raise ValueError(
                    "Must provide a dictionary of inputs for a DAG with multiple entry "
                    "points."
                )
            X = {self._roots[0].name: X}

        return X

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
        self._validate_graph()
        fit_params_steps = self._check_fit_params(**fit_params)
        Xts = self._fit(X, y, **fit_params_steps)
        for leaf in self._leaves:
            with _print_elapsed_time("DAG", self._log_message(leaf)):
                if leaf.estimator != "passthrough":
                    fit_params_leaf = fit_params_steps[leaf.name]
                    deps = self.graph_.predecessors(leaf.name)
                    Xt = _stack([Xts[dep] for dep in deps], axis=leaf.axis)
                    leaf.estimator.fit(Xt, y, **fit_params_leaf)

        return self

    def _check_fit_params(self, **fit_params):
        fit_params_steps = {step.name: {} for step in self._steps if step is not None}
        for pname, pval in fit_params.items():
            if "__" not in pname:
                raise ValueError(
                    f"DAG.fit does not accept the {pname} parameter. "
                    "You can pass parameters to specific steps of your "
                    "DAG using the stepname__parameter format, e.g. "
                    "`DAG.fit(X, y, logisticregression__sample_weight"
                    "=sample_weight)`."
                )
            step, param = pname.split("__", 1)
            fit_params_steps[step][param] = pval
        return fit_params_steps

    def _validate_graph(self):
        self.graph_ = self.graph.copy(as_view=True)
        if len(self.graph_) == 0:
            raise ValueError("DAG has no nodes.")

        self._steps = []
        for i, name in enumerate(nx.topological_sort(self.graph_)):
            step = self.graph_.nodes[name]["step"]
            step.index = i
            self._steps.append(step)

        # validate names
        self._validate_names([step.name for step in self._steps])

        # validate estimators
        self._roots = []
        self._branches = []
        self._leaves = []
        for step in self._steps:
            step.is_root = step.is_leaf = False
            if (
                self.graph_.in_degree(step.name) > 0
                and self.graph_.out_degree(step.name) > 0
            ):
                self._branches.append(step)
            else:
                # Edge case: single estimator. Node can be both a root and a leaf.
                if self.graph_.in_degree(step.name) == 0:
                    step.is_root = True
                    self._roots.append(step)
                if self.graph_.out_degree(step.name) == 0:
                    step.is_leaf = True
                    self._leaves.append(step)

        for step in self._roots + self._branches:
            est = step.estimator
            # Unlike pipelines we also allow predictors to be used as a transformer, to support
            # model stacking.
            if (
                not _is_passthrough(est)
                and not _is_transformer(est)
                and not _is_predictor(est)
            ):
                raise TypeError(
                    "All intermediate steps should be "
                    "transformers and implement fit and transform "
                    "or be the string 'passthrough' "
                    f"'{est}' (type {type(est)}) doesn't"
                )

        for step in self._leaves:
            est = step.estimator
            if not _is_passthrough(est) and not hasattr(est, "fit"):
                raise TypeError(
                    "Leaf nodes of a DAG should implement fit "
                    "or be the string 'passthrough'. "
                    f"'{est}' (type {type(est)}) doesn't"
                )

    def _repr_svg_(self):
        return DAGRenderer(self.graph).draw(format="svg")

    def _repr_png_(self):
        return DAGRenderer(self.graph).draw(format="png")

    def _repr_jpeg_(self):
        return DAGRenderer(self.graph).draw(format="jpg")

    def _repr_html_(self):
        return DAGRenderer(self.graph).draw(format="svg")

    def _repr_pretty_(self, p, cycle):
        if cycle:
            p.text(repr(self))
        else:
            p.text(str(self))

    def _repr_mimebundle_(self, **kwargs):
        # Don't render yet...
        renderers = {
            "image/svg+xml": self._repr_svg_,
            "image/png": self._repr_png_,
            "image/jpeg": self._repr_jpeg_,
            "text/plain": self.__str__,
            "text/html": self._repr_html_,
        }

        include = kwargs.get("include")
        if include:
            renderers = {k: v for k, v in renderers.items() if k in include}

        exclude = kwargs.get("exclude")
        if exclude:
            renderers = {k: v for k, v in renderers.items() if k not in exclude}

        # Now render any remaining options.
        return {k: v() for k, v in renderers.items()}

    def __str__(self):
        return DAGRenderer(self.graph).draw(format="ascii")

    @property
    def nodes(self):
        return self.graph.nodes

    @property
    def edges(self):
        return self.graph.edges
