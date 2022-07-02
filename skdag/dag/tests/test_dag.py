"""
Test the DAG module.
"""
import time
import re

import pytest
import numpy as np
from skdag import DAG, DAGBuilder

from sklearn.utils._testing import assert_array_almost_equal
from sklearn.base import clone, BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

iris = load_iris()

JUNK_FOOD_DOCS = (
    "the pizza pizza beer copyright",
    "the pizza burger beer copyright",
    "the the pizza beer beer copyright",
    "the burger beer beer copyright",
    "the coke burger coke copyright",
    "the coke burger burger",
)


class NoFit:
    """Small class to test parameter dispatching."""

    def __init__(self, a=None, b=None):
        self.a = a
        self.b = b


class NoTrans(NoFit):
    def fit(self, X, y):
        return self

    def get_params(self, deep=False):
        return {"a": self.a, "b": self.b}

    def set_params(self, **params):
        self.a = params["a"]
        return self


class NoInvTransf(NoTrans):
    def transform(self, X):
        return X


class Transf(NoInvTransf):
    def transform(self, X):
        return X

    def inverse_transform(self, X):
        return X


class TransfFitParams(Transf):
    def fit(self, X, y, **fit_params):
        self.fit_params = fit_params
        return self


class Mult(BaseEstimator):
    def __init__(self, mult=1):
        self.mult = mult

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.asarray(X) * self.mult

    def inverse_transform(self, X):
        return np.asarray(X) / self.mult

    def predict(self, X):
        return (np.asarray(X) * self.mult).sum(axis=1)

    predict_proba = predict_log_proba = decision_function = predict

    def score(self, X, y=None):
        return np.sum(X)


class FitParamT(BaseEstimator):
    """Mock classifier"""

    def __init__(self):
        self.successful = False

    def fit(self, X, y, should_succeed=False):
        self.successful = should_succeed
        return self

    def predict(self, X):
        return self.successful

    def fit_predict(self, X, y, should_succeed=False):
        self.fit(X, y, should_succeed=should_succeed)
        return self.predict(X)

    def score(self, X, y=None, sample_weight=None):
        if sample_weight is not None:
            X = X * sample_weight
        return np.sum(X)


class DummyTransf(Transf):
    """Transformer which store the column means"""

    def fit(self, X, y):
        self.means_ = np.mean(X, axis=0)
        # store timestamp to figure out whether the result of 'fit' has been
        # cached or not
        self.timestamp_ = time.time()
        return self


class DummyEstimatorParams(BaseEstimator):
    """Mock classifier that takes params on predict"""

    def fit(self, X, y):
        return self

    def predict(self, X, got_attribute=False):
        self.got_attribute = got_attribute
        return self

    def predict_proba(self, X, got_attribute=False):
        self.got_attribute = got_attribute
        return self

    def predict_log_proba(self, X, got_attribute=False):
        self.got_attribute = got_attribute
        return self


def test_dag_invalid_parameters():
    # Test the various init parameters of the dag in fit
    # method
    dag = DAG.from_pipeline([(1, 1)])
    with pytest.raises(TypeError):
        dag.fit([[1]], [1])

    # Check that we can't fit DAGs with objects without fit
    # method
    msg = (
        "Leaf nodes of a DAG should implement fit or be the string 'passthrough'"
        ".*NoFit.*"
    )
    dag = DAG.from_pipeline([("clf", NoFit())])
    with pytest.raises(TypeError, match=msg):
        dag.fit([[1]], [1])

    # Smoke test with only an estimator
    clf = NoTrans()
    dag = DAG.from_pipeline([("svc", clf)])
    assert dag.get_params(deep=True) == dict(
        svc__a=None, svc__b=None, svc=clf, **dag.get_params(deep=False)
    )

    # Check that params are set
    dag.set_params(svc__a=0.1)
    assert clf.a == 0.1
    assert clf.b is None
    # Smoke test the repr:
    repr(dag)

    # Test with two objects
    clf = SVC()
    filter1 = SelectKBest(f_classif)
    dag = DAG.from_pipeline([("anova", filter1), ("svc", clf)])

    # Check that estimators are not cloned on pipeline construction
    assert dag.named_steps["anova"] is filter1
    assert dag.named_steps["svc"] is clf

    # Check that we can't fit with non-transformers on the way
    # Note that NoTrans implements fit, but not transform
    msg = "All intermediate steps should be transformers.*\\bNoTrans\\b.*"
    dag2 = DAG.from_pipeline([("t", NoTrans()), ("svc", clf)])
    with pytest.raises(TypeError, match=msg):
        dag2.fit([[1]], [1])

    # Check that params are set
    dag.set_params(svc__C=0.1)
    assert clf.C == 0.1
    # Smoke test the repr:
    repr(dag)

    # Check that params are not set when naming them wrong
    msg = re.escape(
        "Invalid parameter 'C' for estimator SelectKBest(). Valid parameters are: ['k',"
        " 'score_func']."
    )
    with pytest.raises(ValueError, match=msg):
        dag.set_params(anova__C=0.1)

    # Test clone
    dag2 = clone(dag)
    assert not dag.named_steps["svc"] is dag2.named_steps["svc"]

    # Check that apart from estimators, the parameters are the same
    params = dag.get_params(deep=True)
    params2 = dag2.get_params(deep=True)

    for x in dag.get_params(deep=False):
        params.pop(x)

    for x in dag.get_params(deep=False):
        params2.pop(x)

    # Remove estimators that where copied
    params.pop("svc")
    params.pop("anova")
    params2.pop("svc")
    params2.pop("anova")
    assert params == params2


def test_dag_pipeline_init():
    # Build a dag from a pipeline
    X = np.array([[1, 2]])
    steps = (("transf", Transf()), ("clf", FitParamT()))
    pipe = Pipeline(steps)
    for inp in [pipe, steps]:
        dag = DAG.from_pipeline(steps)
        dag.fit(X, y=None)
        dag.score(X)

        dag.set_params(transf="passthrough")
        dag.fit(X, y=None)
        dag.score(X)


def test_dag_methods_anova():
    # Test the various methods of the dag (anova).
    X = iris.data
    y = iris.target
    # Test with Anova + LogisticRegression
    clf = LogisticRegression()
    filter1 = SelectKBest(f_classif, k=2)
    dag1 = DAG.from_pipeline([("anova", filter1), ("logistic", clf)])
    dag2 = (
        DAGBuilder()
        .add_step("anova", filter1)
        .add_step("logistic", clf, deps=["anova"])
        .make_dag()
    )
    dag1.fit(X, y)
    dag2.fit(X, y)
    assert_array_almost_equal(dag1.predict(X), dag2.predict(X))
    assert_array_almost_equal(dag1.predict_proba(X), dag2.predict_proba(X))
    assert_array_almost_equal(dag1.predict_log_proba(X), dag2.predict_log_proba(X))
    assert_array_almost_equal(dag1.score(X, y), dag2.score(X, y))


def test_dag_fit_params():
    # Test that the pipeline can take fit parameters
    dag = DAG.from_pipeline([("transf", Transf()), ("clf", FitParamT())])
    dag.fit(X=None, y=None, clf__should_succeed=True)
    # classifier should return True
    assert dag.predict(None)
    # and transformer params should not be changed
    assert dag.named_steps["transf"].a is None
    assert dag.named_steps["transf"].b is None
    # invalid parameters should raise an error message

    msg = re.escape("fit() got an unexpected keyword argument 'bad'")
    with pytest.raises(TypeError, match=msg):
        dag.fit(None, None, clf__bad=True)


def test_dag_sample_weight_supported():
    # DAG should pass sample_weight
    X = np.array([[1, 2]])
    dag = DAG.from_pipeline([("transf", Transf()), ("clf", FitParamT())])
    dag.fit(X, y=None)
    assert dag.score(X) == 3
    assert dag.score(X, y=None) == 3
    assert dag.score(X, y=None, sample_weight=None) == 3
    assert dag.score(X, sample_weight=np.array([2, 3])) == 8


def test_dag_sample_weight_unsupported():
    # When sample_weight is None it shouldn't be passed
    X = np.array([[1, 2]])
    dag = DAG.from_pipeline([("transf", Transf()), ("clf", Mult())])
    dag.fit(X, y=None)
    assert dag.score(X) == 3
    assert dag.score(X, sample_weight=None) == 3

    msg = re.escape("score() got an unexpected keyword argument 'sample_weight'")
    with pytest.raises(TypeError, match=msg):
        dag.score(X, sample_weight=np.array([2, 3]))


def test_dag_raise_set_params_error():
    # Test dag raises set params error message for nested models.
    dag = DAG.from_pipeline([("cls", LinearRegression())])

    # expected error message
    error_msg = (
        r"Invalid parameter 'fake' for estimator DAG\(graph=<networkx\..*DiGraph[^>]*>"
        r"\)\. Valid parameters are: \['graph', 'memory', 'n_jobs', 'verbose'\]."
    )
    with pytest.raises(ValueError, match=error_msg):
        dag.set_params(fake="nope")

    # invalid outer parameter name for compound parameter: the expected error message
    # is the same as above.
    with pytest.raises(ValueError, match=error_msg):
        dag.set_params(fake__estimator="nope")

    # expected error message for invalid inner parameter
    error_msg = (
        r"Invalid parameter 'invalid_param' for estimator LinearRegression\(\)\. Valid"
        r" parameters are: \['copy_X', 'fit_intercept', 'n_jobs', 'normalize',"
        r" 'positive'\]."
    )
    with pytest.raises(ValueError, match=error_msg):
        dag.set_params(cls__invalid_param="nope")


def test_dag_stacking_pca_svm_rf():
    # Test the various methods of the pipeline (pca + svm).
    X = iris.data
    y = iris.target
    # Build a simple model stack with some preprocessing.
    pca = PCA(svd_solver="full", n_components="mle", whiten=True)
    svc = SVC(probability=True, random_state=0)
    rf = RandomForestClassifier(random_state=0)
    log = LogisticRegression()

    dag = (
        DAGBuilder()
        .add_step("pca", pca)
        .add_step("svc", svc, deps=["pca"])
        .add_step("rf", rf, deps=["pca"])
        .add_step("log", log, deps=["svc", "rf"])
        .make_dag()
    )
    dag.fit(X, y)

    prob_shape = len(iris.target), len(iris.target_names)
    tgt_shape = iris.target.shape

    assert dag.predict_proba(X).shape == prob_shape
    assert dag.predict(X).shape == tgt_shape
    assert dag.predict_log_proba(X).shape == prob_shape
    assert isinstance(dag.score(X, y), (float, np.floating))
