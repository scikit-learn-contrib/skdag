"""
Directed Acyclic Graphs (DAGs) may be used to construct complex workflows for
scikit-learn estimators. As the name suggests, data may only flow in one
direction and can't go back on itself to a previously run step.
"""
import networkx


class DAG:
    """
    A Directed Acyclic Graph (DAG) for scikit-learn estimator workflows.
    """
    pass
