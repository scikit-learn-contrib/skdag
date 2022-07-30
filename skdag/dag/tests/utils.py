from time import time

import numpy as np
from sklearn.base import BaseEstimator


class NoFit:  # pragma: no cover
    """Small class to test parameter dispatching."""

    def __init__(self, a=None, b=None):
        self.a = a
        self.b = b


class NoTrans(NoFit):  # pragma: no cover
    def fit(self, X, y):
        return self

    def get_params(self, deep=False):
        return {"a": self.a, "b": self.b}

    def set_params(self, **params):
        self.a = params["a"]
        return self


class NoInvTransf(NoTrans):  # pragma: no cover
    def transform(self, X):
        return X


class Transf(NoInvTransf):  # pragma: no cover
    def inverse_transform(self, X):
        return X


class TransfFitParams(Transf):  # pragma: no cover
    def fit(self, X, y, **fit_params):
        self.fit_params = fit_params
        return self


class Mult(BaseEstimator):  # pragma: no cover
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


class FitParamT(BaseEstimator):  # pragma: no cover
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


class DummyTransf(Transf):  # pragma: no cover
    """Transformer which store the column means"""

    def fit(self, X, y):
        self.means_ = np.mean(X, axis=0)
        # store timestamp to figure out whether the result of 'fit' has been
        # cached or not
        self.timestamp_ = time()
        return self


class DummyEstimatorParams(BaseEstimator):  # pragma: no cover
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
