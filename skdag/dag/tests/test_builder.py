import pytest
from skdag import DAG, DAGBuilder
from skdag.dag.tests.utils import Mult, Transf
from skdag.exceptions import DAGError
from sklearn.pipeline import Pipeline


def test_builder_basics():
    builder = DAGBuilder()
    builder.add_step("tr", Transf())
    builder.add_step("ident", "passthrough", deps=["tr"])
    builder.add_step("est", Mult(), deps=["ident"])

    dag = builder.make_dag()
    assert isinstance(dag, DAG)
    assert dag._repr_html_() == builder._repr_html_()

    with pytest.raises(ValueError):
        builder.add_step("est2", Mult(), deps=["missing"])

    with pytest.raises(ValueError):
        builder.add_step("est2", Mult(), deps=[1])

    with pytest.raises(KeyError):
        # Step names *must* be strings.
        builder.add_step(2, Mult())

    with pytest.raises(KeyError):
        # Step names *must* be unique.
        builder.add_step("est", Mult())

    with pytest.raises(DAGError):
        # Cycles not allowed.
        builder.graph.add_edge("est", "ident")
        builder.make_dag()


def test_pipeline():
    steps = [("tr", Transf()), ("ident", "passthrough"), ("est", Mult())]

    builder = DAGBuilder()
    prev = None
    for name, est in steps:
        builder.add_step(name, est, deps=[prev] if prev else None)
        prev = name

    dag = builder.make_dag()

    assert dag.steps_ == steps
