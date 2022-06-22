import pytest

from sklearn.datasets import load_iris


@pytest.fixture
def data():
    return load_iris(return_X_y=True)
