"""
Directed Acyclic Graphs (DAGs) may be used to construct complex workflows for
scikit-learn estimators. As the name suggests, data may only flow in one
direction and can't go back on itself to a previously run step.
"""
from collections import UserDict

import networkx as nx
from joblib import Parallel, delayed
from skdag.dag._render import DAGRenderer
from skdag.dag._utils import _is_passthrough, _is_predictor, _is_transformer, _stack
from sklearn.base import clone
from sklearn.utils import Bunch, _print_elapsed_time, _safe_indexing
from sklearn.utils.metaestimators import _BaseComposition, available_if
from sklearn.utils.validation import check_memory

__all__ = ["DAG", "DAGStep"]


def _transform_one(transformer, X, weight, allow_predictor=True, **fit_params):
    if _is_passthrough(transformer):
        res = X
    elif allow_predictor and hasattr(transformer, "predict"):
        res = transformer.predict(X)
        if res.ndim < 2:
            res = res.reshape(-1, 1)
    else:
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
        if _is_passthrough(transformer):
            res = X
        elif hasattr(transformer, "fit_transform"):
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


def _leaf_estimators_have(attr, how="all"):
    """Check that leaves have `attr`.
    Used together with `avaliable_if` in `DAG`."""

    def check(self):
        # raises `AttributeError` with all details if `attr` does not exist
        failed = []
        for leaf in self.leaves_:
            try:
                (_is_passthrough(leaf.estimator) and attr == "transform") or getattr(
                    leaf.estimator, attr
                )
            except AttributeError:
                failed.append(leaf.estimator)

        if (how == "all" and failed) or (
            how == "any" and len(failed) != len(self.leaves_)
        ):
            raise AttributeError(
                f"{', '.join([repr(type(est)) for est in failed])} "
                f"object(s) has no attribute '{attr}'"
            )
        return True

    return check


def _parallel_fit(dag, step, Xin, Xs, y, fit_transform_fn, memory, **fit_params):
    transformer = step.estimator

    if step.deps:
        X = _stack(
            [_safe_indexing(Xs[dep], step.deps[dep], axis=1) for dep in step.deps],
            axis=step.axis,
        )
    else:
        # For root nodes, the destination rather than the source is
        # specified.
        X = Xin[step.name]

    clsname = type(dag).__name__
    if transformer is None or transformer == "passthrough":
        with _print_elapsed_time(clsname, dag._log_message(step)):
            return X, transformer

    if hasattr(memory, "location") and memory.location is None:
        # we do not clone when caching is disabled to
        # preserve backward compatibility
        cloned_transformer = transformer
    else:
        cloned_transformer = clone(transformer)

    # Fit or load from cache the current transformer
    Xt, fitted_transformer = fit_transform_fn(
        cloned_transformer,
        X,
        y,
        None,
        message_clsname=clsname,
        message=dag._log_message(step),
        **fit_params,
    )

    return Xt, fitted_transformer


def _parallel_transform(dag, step, Xin, Xs, transform_fn, **fn_params):
    transformer = step.estimator
    if step.deps:
        X = _stack(
            [_safe_indexing(Xs[dep], step.deps[dep], axis=1) for dep in step.deps],
            axis=step.axis,
        )
    else:
        # For root nodes, the destination rather than the source is
        # specified.
        X = Xin[step.name]

    clsname = type(dag).__name__
    if transformer is None or transformer == "passthrough":
        with _print_elapsed_time(clsname, dag._log_message(step)):
            return X

    # Fit or load from cache the current transformer
    Xt = transform_fn(
        transformer,
        X,
        None,
        message_clsname=clsname,
        message=dag._log_message(step),
        **fn_params,
    )

    return Xt


def _parallel_fit_leaf(dag, leaf, Xts, y, **fit_params):
    with _print_elapsed_time(type(dag).__name__, dag._log_message(leaf)):
        if leaf.estimator == "passthrough":
            fitted_estimator = leaf.estimator
        else:
            Xt = _stack([Xts[dep] for dep in leaf.deps], axis=leaf.axis)
            fitted_estimator = leaf.estimator.fit(Xt, y, **fit_params)

    return fitted_estimator


def _parallel_execute(
    dag, leaf, fn, Xts, y=None, fit_first=False, fit_params=None, fn_params=None
):
    with _print_elapsed_time("DAG", dag._log_message(leaf)):
        Xt = _stack(
            [_safe_indexing(Xts[dep], leaf.deps[dep], axis=1) for dep in leaf.deps],
            axis=leaf.axis,
        )
        fit_params = fit_params or {}
        fn_params = fn_params or {}
        if leaf.estimator == "passthrough":
            Xout = Xt
        elif fit_first:
            if hasattr(leaf.estimator, f"fit_{fn}"):
                Xout = getattr(leaf.estimator, f"fit_{fn}")(Xt, y, **fit_params)
            else:
                leaf.estimator.fit(Xt, y, **fit_params)
                Xout = getattr(leaf.estimator, fn)(Xt, **fn_params)
        else:
            Xout = getattr(leaf.estimator, fn)(Xt, **fn_params)

    fitted_estimator = leaf.estimator
    return Xout, fitted_estimator


class DAGStep:
    """
    A single estimator step in a DAG.

    Parameters
    ----------
    name : str
        The reference name for this step.
    estimator : estimator-like
        The estimator (transformer or predictor) that will be executed by this step.
    deps : dict
        A map of dependency names to columns. If columns is ``None``, then all input
        columns will be selected.
    axis : int, default = 1
        The strategy for merging inputs if there is more than upstream dependency.
        ``axis=0`` will assume all inputs have the same features and stack the rows
        together; ``axis=1`` will assume each input provides different features for the
        same samples.
    """

    def __init__(self, name, estimator, deps, axis=1):
        self.name = name
        self.estimator = estimator
        self.deps = deps
        self.axis = axis
        self.is_root = False
        self.is_leaf = False
        self.is_fitted = False

        self._hashval = hash((type(self).__name__, self.name))

    def __repr__(self):
        return f"{type(self).__name__}({repr(self.name)}, {repr(self.estimator)})"

    def __hash__(self):
        return self._hashval


class DAG(_BaseComposition):
    """
    A Directed Acyclic Graph (DAG) of estimators, that itself implements the estimator
    interface.

    A DAG may consist of a simple chain of estimators (being exactly equivalent to a
    :mod:`sklearn.pipeline.Pipeline`) or a more complex path of dependencies. But as the
    name suggests, it may not contain any cyclic dependencies and data may only flow
    from one or more start points (roots) to one or more endpoints (leaves).

    The simplest DAGs are just a chain of singular dependencies. These DAGs may be
    created from the :meth:`skdag.dag.DAG.from_pipeline` method in the same way as a
    DAG:

    >>> from sklearn.decomposition import PCA
    >>> from sklearn.impute import SimpleImputer
    >>> from sklearn.linear_model import LogisticRegression
    >>> dag = DAG.from_pipeline(
    ...     steps=[
    ...         ("impute", SimpleImputer()),
    ...         ("pca", PCA()),
    ...         ("lr", LogisticRegression())
    ...     ]
    ... )
    >>> print(dag)
    o    impute
    |
    o    pca
    |
    o    lr

    For more complex DAGs, it is recommended to use a :class:`skdag.dag.DAGBuilder`,
    which allows you to define the graph by specifying the dependencies of each new
    estimator:

    >>> dag = (
    ...     DAGBuilder()
    ...     .add_step("impute", SimpleImputer())
    ...     .add_step("vitals", "passthrough", deps={"imp": slice(0, 4)})
    ...     .add_step("blood", PCA(n_components=2, random_state=0), deps={"imp": slice(4, 10)})
    ...     .add_step("rf", RandomForestRegressor(random_state=0), deps=["blood", "vitals"])
    ...     .make_dag()
    ... )
    o    impute
    |\
    o o    blood,vitals
    |/
    o    rf

    In the above examples we pass the first four columns directly to a regressor, but
    the remaining columns have dimensionality reduction applied first before being
    passed to the same regressor. Note that we can define our graph edges in two
    different ways: as a dict (if we need to select only certain columns from the source
    node) or as a simple list (if we want to simply grab all columns from all input
    nodes).

    The DAG may now be used as an estimator in its own right:

    >>> from sklearn import datasets
    >>> X, y = datasets.load_diabetes(return_X_y=True)
    >>> y_hat = cmplx.fit_predict(X, y)
    ... array([...

    Parameters
    ----------

    graph : networkx.DiGraph
        A DAG of :class:`skdag.dag.DAGStep` defining the workflow.

    memory : str or object with the joblib.Memory interface, default=None
        Used to cache the fitted transformers of the DAG. By default,
        no caching is performed. If a string is given, it is the path to
        the caching directory. Enabling caching triggers a clone of
        the transformers before fitting. Therefore, the transformer
        instance given to the DAG cannot be inspected
        directly. Use the attribute ``named_steps`` or ``steps`` to
        inspect estimators within the DAG. Caching the
        transformers is advantageous when fitting is time consuming.

    verbose : bool, default=False
        If True, the time elapsed while fitting each step will be printed as it
        is completed.

    n_jobs : int, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
    """

    # BaseEstimator interface
    _required_parameters = ["graph"]

    @classmethod
    def from_pipeline(cls, steps, **kwargs):
        graph = nx.DiGraph()
        if hasattr(steps, "steps"):
            DAG = steps
            steps = DAG.steps
            if hasattr(DAG, "get_params"):
                kwargs = {
                    **{
                        k: v
                        for k, v in DAG.get_params().items()
                        if k in ("memory", "verbose")
                    },
                    **kwargs,
                }

        for i in range(len(steps)):
            name, estimator = steps[i]
            deps = {}
            if i > 0:
                dep = steps[i - 1][0]
                deps[dep] = None

            step = DAGStep(name, estimator, deps)
            graph.add_node(name, step=step)
            if deps:
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
        estimators contained within the `steps_` of the `DAG`.
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
        return self._get_params("steps_", deep=deep)

    def set_params(self, **kwargs):
        """Set the parameters of this estimator.
        Valid parameter keys can be listed with ``get_params()``. Note that
        you can directly set the parameters of the estimators contained in
        `steps_`.
        Parameters
        ----------
        **kwargs : dict
            Parameters of this estimator or parameters of estimators contained
            in `steps`. Parameters of the steps may be set using its name and
            the parameter name separated by a '__'.
        Returns
        -------
        self : object
            DAG class instance.
        """
        self._set_params("steps_", **kwargs)
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
        return len(self.graph_)

    def __getitem__(self, name):
        """
        Retrieve a named estimator.
        """
        return self.graph_.nodes[name]["step"].estimator

    def _fit(self, X, y=None, **fit_params_steps):
        # Setup the memory
        memory = check_memory(self.memory)

        fit_transform_one_cached = memory.cache(_fit_transform_one)

        Xin = self._resolve_inputs(X)
        Xs = {}
        with Parallel(n_jobs=self.n_jobs) as parallel:
            for stage in self._iter(with_leaves=False, filter_passthrough=False):
                stage_names = [step.name for step in stage]
                outputs, fitted_transformers = zip(
                    *parallel(
                        delayed(_parallel_fit)(
                            self,
                            step,
                            Xin,
                            Xs,
                            y,
                            fit_transform_one_cached,
                            memory,
                            **fit_params_steps[step.name],
                        )
                        for step in stage
                    )
                )

                for step, fitted_transformer in zip(stage, fitted_transformers):
                    # Replace the transformer of the step with the fitted
                    # transformer. This is necessary when loading the transformer
                    # from the cache.
                    step.estimator = fitted_transformer
                    step.is_fitted = True

                Xs.update(dict(zip(stage_names, outputs)))

                # If all of a dep's dependents are now complete, we can free up some
                # memory.
                for dep in {dep for step in stage for dep in step.deps}:
                    dependents = self.graph_.successors(dep)
                    if all(d in Xs for d in dependents):
                        del Xs[dep]

        return Xs

    def _transform(self, X, **fn_params_steps):
        # Setup the memory
        memory = check_memory(self.memory)

        transform_one_cached = memory.cache(_transform_one)

        Xin = self._resolve_inputs(X)
        Xs = {}
        with Parallel(n_jobs=self.n_jobs) as parallel:
            for stage in self._iter(with_leaves=False, filter_passthrough=False):
                stage_names = [step.name for step in stage]
                outputs = parallel(
                    delayed(_parallel_transform)(
                        self,
                        step,
                        Xin,
                        Xs,
                        transform_one_cached,
                        **fn_params_steps[step.name],
                    )
                    for step in stage
                )

                Xs.update(dict(zip(stage_names, outputs)))

                # If all of a dep's dependents are now complete, we can free up some
                # memory.
                for dep in {dep for step in stage for dep in step.deps}:
                    dependents = self.graph_.successors(dep)
                    if all(d in Xs for d in dependents):
                        del Xs[dep]

        return Xs

    def _resolve_inputs(self, X):
        if isinstance(X, (dict, UserDict)):
            inputs = sorted(X.keys())
            if inputs != sorted(root.name for root in self.roots_):
                raise ValueError(
                    "Input dicts must contain one key per entry node. "
                    f"Entry nodes are {self.roots_}, got {inputs}."
                )
        else:
            if len(self.roots_) != 1:
                raise ValueError(
                    "Must provide a dictionary of inputs for a DAG with multiple entry "
                    "points."
                )
            X = {self.roots_[0].name: X}

        return X

    def _match_input_format(self, Xin, Xout):
        if len(self.leaves_) == 1 and not isinstance(Xin, (dict, UserDict)):
            return Xout[self.leaves_[0].name]
        return Bunch(**Xout)

    def fit(self, X, y=None, **fit_params):
        """Fit the model.
        Fit all the transformers one after the other and transform the
        data. Finally, fit the transformed data using the final estimator.
        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of the
            DAG.
        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps of
            the DAG.
        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.
        Returns
        -------
        self : object
            DAG fitted steps.
        """
        self._validate_graph()
        fit_params_steps = self._check_fit_params(**fit_params)
        Xts = self._fit(X, y, **fit_params_steps)
        fitted_estimators = Parallel(n_jobs=self.n_jobs)(
            [
                delayed(_parallel_fit_leaf)(
                    self, leaf, Xts, y, **fit_params_steps[leaf.name]
                )
                for leaf in self.leaves_
            ]
        )
        for est, leaf in zip(fitted_estimators, self.leaves_):
            leaf.estimator = est
            leaf.is_fitted = True

        return self

    def _fit_execute(self, fn, X, y=None, **fit_params):
        self._validate_graph()
        fit_params_steps = self._check_fit_params(**fit_params)
        Xts = self._fit(X, y, **fit_params_steps)
        Xout = {}

        leaf_names = [leaf.name for leaf in self.leaves_]
        outputs, fitted_estimators = zip(
            *Parallel(n_jobs=self.n_jobs)(
                delayed(_parallel_execute)(
                    self,
                    leaf,
                    fn,
                    Xts,
                    y,
                    fit_first=True,
                    fit_params=fit_params_steps[leaf.name],
                )
                for leaf in self.leaves_
            )
        )

        Xout = dict(zip(leaf_names, outputs))
        for step, fitted_estimator in zip(self.leaves_, fitted_estimators):
            step.estimator = fitted_estimator
            step.is_fitted = True

        return self._match_input_format(X, Xout)

    def _execute(self, fn, X, **fn_params):
        Xout = {}
        fn_params_steps = self._check_fn_params(**fn_params)
        Xts = self._transform(X, **fn_params_steps)

        leaf_names = [leaf.name for leaf in self.leaves_]
        outputs, _ = zip(
            *Parallel(n_jobs=self.n_jobs)(
                delayed(_parallel_execute)(
                    self,
                    leaf,
                    fn,
                    Xts,
                    fit_first=False,
                    fn_params=fn_params_steps[leaf.name],
                )
                for leaf in self.leaves_
            )
        )

        Xout = dict(zip(leaf_names, outputs))

        return self._match_input_format(X, Xout)

    @available_if(_leaf_estimators_have("transform"))
    def fit_transform(self, X, y=None, **fit_params):
        """Fit the model and transform with the final estimator.
        Fits all the transformers one after the other and transform the
        data. Then uses `fit_transform` on transformed data with the final
        estimator.
        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of the
            DAG.
        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps of
            the DAG.
        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.
        Returns
        -------
        Xt : ndarray of shape (n_samples, n_transformed_features)
            Transformed samples.
        """
        return self._fit_execute("transform", X, y, **fit_params)

    @available_if(_leaf_estimators_have("transform"))
    def transform(self, X):
        """Transform the data, and apply `transform` with the final estimator.
        Call `transform` of each transformer in the DAG. The transformed
        data are finally passed to the final estimator that calls
        `transform` method. Only valid if the final estimator
        implements `transform`.
        This also works where final estimator is `None` in which case all prior
        transformations are applied.
        Parameters
        ----------
        X : iterable
            Data to transform. Must fulfill input requirements of first step
            of the DAG.
        Returns
        -------
        Xt : ndarray of shape (n_samples, n_transformed_features)
            Transformed data.
        """
        return self._execute("transform", X)

    @available_if(_leaf_estimators_have("predict"))
    def fit_predict(self, X, y=None, **fit_params):
        """Transform the data, and apply `fit_predict` with the final estimator.
        Call `fit_transform` of each transformer in the DAG. The
        transformed data are finally passed to the final estimator that calls
        `fit_predict` method. Only valid if the final estimator implements
        `fit_predict`.
        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of
            the DAG.
        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps
            of the DAG.
        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.
        Returns
        -------
        y_pred : ndarray
            Result of calling `fit_predict` on the final estimator.
        """
        return self._fit_execute("predict", X, y, **fit_params)

    @available_if(_leaf_estimators_have("predict"))
    def predict(self, X, **predict_params):
        """Transform the data, and apply `predict` with the final estimator.
        Call `transform` of each transformer in the DAG. The transformed
        data are finally passed to the final estimator that calls `predict`
        method. Only valid if the final estimator implements `predict`.
        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the DAG.
        **predict_params : dict of string -> object
            Parameters to the ``predict`` called at the end of all
            transformations in the DAG. Note that while this may be
            used to return uncertainties from some models with return_std
            or return_cov, uncertainties that are generated by the
            transformations in the DAG are not propagated to the
            final estimator.
            .. versionadded:: 0.20
        Returns
        -------
        y_pred : ndarray
            Result of calling `predict` on the final estimator.
        """
        return self._execute("predict", X, **predict_params)

    @available_if(_leaf_estimators_have("predict_proba"))
    def predict_proba(self, X, **predict_proba_params):
        """Transform the data, and apply `predict_proba` with the final estimator.
        Call `transform` of each transformer in the DAG. The transformed
        data are finally passed to the final estimator that calls
        `predict_proba` method. Only valid if the final estimator implements
        `predict_proba`.
        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the DAG.
        **predict_proba_params : dict of string -> object
            Parameters to the `predict_proba` called at the end of all
            transformations in the DAG.
        Returns
        -------
        y_proba : ndarray of shape (n_samples, n_classes)
            Result of calling `predict_proba` on the final estimator.
        """
        return self._execute("predict_proba", X, **predict_proba_params)

    @available_if(_leaf_estimators_have("decision_function"))
    def decision_function(self, X):
        """Transform the data, and apply `decision_function` with the final estimator.
        Call `transform` of each transformer in the DAG. The transformed
        data are finally passed to the final estimator that calls
        `decision_function` method. Only valid if the final estimator
        implements `decision_function`.
        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the DAG.
        Returns
        -------
        y_score : ndarray of shape (n_samples, n_classes)
            Result of calling `decision_function` on the final estimator.
        """
        return self._execute("decision_function", X)

    @available_if(_leaf_estimators_have("score_samples"))
    def score_samples(self, X):
        """Transform the data, and apply `score_samples` with the final estimator.
        Call `transform` of each transformer in the DAG. The transformed
        data are finally passed to the final estimator that calls
        `score_samples` method. Only valid if the final estimator implements
        `score_samples`.
        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the DAG.
        Returns
        -------
        y_score : ndarray of shape (n_samples,)
            Result of calling `score_samples` on the final estimator.
        """
        return self._execute("score_samples", X)

    @available_if(_leaf_estimators_have("predict_log_proba"))
    def predict_log_proba(self, X, **predict_log_proba_params):
        """Transform the data, and apply `predict_log_proba` with the final estimator.
        Call `transform` of each transformer in the DAG. The transformed
        data are finally passed to the final estimator that calls
        `predict_log_proba` method. Only valid if the final estimator
        implements `predict_log_proba`.
        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the DAG.
        **predict_log_proba_params : dict of string -> object
            Parameters to the ``predict_log_proba`` called at the end of all
            transformations in the DAG.
        Returns
        -------
        y_log_proba : ndarray of shape (n_samples, n_classes)
            Result of calling `predict_log_proba` on the final estimator.
        """
        return self._execute("predict_log_proba", X, **predict_log_proba_params)

    def _check_fit_params(self, **fit_params):
        fit_params_steps = {
            name: {} for (name, step) in self.steps_ if step is not None
        }
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

    def _check_fn_params(self, **fn_params):
        global_params = {}
        fn_params_steps = {name: {} for (name, step) in self.steps_ if step is not None}
        for pname, pval in fn_params.items():
            if "__" not in pname:
                global_params[pname] = pval
            step, param = pname.split("__", 1)
            fn_params_steps[step][param] = pval

        for step in fn_params_steps:
            fn_params_steps.update(global_params)

        return fn_params_steps

    def _validate_graph(self):
        if len(self.graph_) == 0:
            raise ValueError("DAG has no nodes.")

        for i, (name, est) in enumerate(self.steps_):
            step = self.graph_.nodes[name]["step"]
            step.index = i

        # validate names
        self._validate_names([name for (name, step) in self.steps_])

        # validate transformers
        for step in self.roots_ + self.branches_:
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

        # Validate final estimator(s)
        for step in self.leaves_:
            est = step.estimator
            if not _is_passthrough(est) and not hasattr(est, "fit"):
                raise TypeError(
                    "Leaf nodes of a DAG should implement fit "
                    "or be the string 'passthrough'. "
                    f"'{est}' (type {type(est)}) doesn't"
                )

    @property
    def graph_(self):
        if not hasattr(self, "_graph"):
            self._graph = self.graph.copy(as_view=True)

        return self._graph

    @property
    def leaves_(self):
        if not hasattr(self, "_leaves"):
            self._leaves = []
            for name, estimator in self.steps_:
                if self.graph_.out_degree(name) == 0:
                    step = self.graph_.nodes[name]["step"]
                    step.is_leaf = True
                    self._leaves.append(step)

        return self._leaves

    @property
    def branches_(self):
        if not hasattr(self, "_branches"):
            self._branches = []
            for name, estimator in self.steps_:
                if self.graph_.in_degree(name) > 0 and self.graph_.out_degree(name) > 0:
                    step = self.graph_.nodes[name]["step"]
                    self._branches.append(step)

        return self._branches

    @property
    def roots_(self):
        if not hasattr(self, "_roots"):
            self._roots = []
            for name, estimator in self.steps_:
                if self.graph_.in_degree(name) == 0:
                    step = self.graph_.nodes[name]["step"]
                    step.is_root = True
                    self._roots.append(step)

        return self._roots

    @property
    def steps_(self):
        "return list of (name, estimator) tuples to conform with Pipeline interface."
        if not hasattr(self, "_steps"):
            self._steps = [
                (node, self.graph_.nodes[node]["step"].estimator)
                for node in nx.topological_sort(self.graph_)
            ]
        return self._steps

    def draw(self, filename=None, style=None, format=None, layout="dot"):
        """
        Render a graphical view of the DAG.

        By default the rendered file will be returned as a string. However if an output
        file is provided then the output will be saved to file.

        Parameters
        ----------

        filename : str
            The file to write the image to.
        style : str, optional, choice of ['light', 'dark']
            Draw the image in light or dark mode.
        format : str, choice of ['svg', 'png', 'jpg', 'txt']
            The rendering format to use. MAy be omitted if the format can be inferred
            from the filename.
        layout : str, default = 'dot'
            The program to use for generating a graph layout.

        Returns
        -------

        output : str, bytes or None
            If a filename is provided the output is written to file and `None` is
            returned. Otherwise, the output is returned as a string (for textual formats
            like ascii or svg) or bytes.
        """
        if format is None:
            format = filename.split(".")[-1]

        if format not in ["svg", "png", "jpg", "txt"]:
            raise ValueError(f"Unsupported file format '{format}'")

        render = DAGRenderer(self.graph_, style=style).draw(
            format=format, layout=layout
        )
        if filename is None:
            return render
        else:
            mode = "wb" if isinstance(render, bytes) else "w"
            with open(filename, mode) as fp:
                fp.write(render)

    def __str__(self):
        return self.draw(format="txt")

    def _repr_svg_(self):
        return self.draw(format="svg")

    def _repr_png_(self):
        return self.draw(format="png")

    def _repr_jpeg_(self):
        return self.draw(format="jpg")

    def _repr_html_(self):
        return self.draw(format="svg")

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

    @property
    def nodes(self):
        return self.graph_.nodes

    @property
    def edges(self):
        return self.graph_.edges