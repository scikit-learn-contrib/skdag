import pytest

from sklearn.utils.estimator_checks import check_estimator

from skdag import TemplateEstimator
from skdag import TemplateClassifier
from skdag import TemplateTransformer


@pytest.mark.parametrize(
    "estimator",
    [TemplateEstimator(), TemplateTransformer(), TemplateClassifier()]
)
def test_all_estimators(estimator):
    return check_estimator(estimator)
