skdag - scikit-learn workflow management
============================================

scikit-dag (``skdag``) is an open-sourced, MIT-licenced library that provides advanced
workflow management to any machine learning operations that follow
:mod:`sklearn` conventions. It does this by introducing Directed Acyclic
Graphs (:class:`skdag.dag.DAG`) as a drop-in replacement for traditional scikit-learn
:mod:`sklearn.pipeline.Pipeline`. This gives you a simple interface for a range of use
cases including complex pre-processing, model stacking and benchmarking.

.. code-block:: python

   from skdag import DAGBuilder

   dag = (
      DAGBuilder(infer_dataframe=True)
      .add_step("impute", SimpleImputer())
      .add_step(
         "vitals",
         "passthrough",
         deps={"impute": ["age", "sex", "bmi", "bp"]},
      )
      .add_step(
         "blood",
         PCA(n_components=2, random_state=0),
         deps={"impute": ["s1", "s2", "s3", "s4", "s5", "s6"]},
      )
      .add_step(
         "rf",
         RandomForestRegressor(max_depth=5, random_state=0),
         deps=["blood", "vitals"],
      )
      .add_step("svm", SVR(C=0.7), deps=["blood", "vitals"])
      .add_step(
         "knn",
         KNeighborsRegressor(n_neighbors=5),
         deps=["blood", "vitals"],
      )
      .add_step("meta", LinearRegression(), deps=["rf", "svm", "knn"])
      .make_dag(n_jobs=2, verbose=True)
   )

   dag.show(detailed=True)

.. image:: _static/img/cover.png

The above DAG imputes missing values, runs PCA on the columns relating to blood test
results and leaves the other columns as they are. Then they get passed to three
different regressors before being passed onto a final meta-estimator. Because DAGs
(unlike pipelines) allow predictors in the middle or a workflow, you can use them to
implement model stacking. We also chose to run the DAG steps in parallel wherever
possible.

After building our DAG, we can treat it as any other estimator:

.. code-block:: python

   from sklearn import datasets

   X, y = datasets.load_diabetes(return_X_y=True, as_frame=True)
   X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=0
   )

   dag.fit(X_train, y_train)
   dag.predict(X_test)

Just like a pipeline, you can optimise it with a gridsearch, pickle it etc.

Note that this package does not deal with things like delayed dependencies and
distributed architectures - consider an `established <https://airflow.apache.org/>`_
`solution <https://dagster.io/>`_ for such use cases. ``skdag`` is just for building and
executing local ensembles from estimators.

:ref:`Read on<quickstart>` to learn more about ``skdag``...

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Getting Started

   quick_start

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Documentation

   user_guide

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: API

   api

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Tutorial - Examples

   auto_examples/index

`Getting started <quick_start.html>`_
-------------------------------------

A practical introduction to DAGs for scikit-learn.

`User Guide <user_guide.html>`_
-------------------------------

Details of the full functionality provided by ``skdag``.

`API Documentation <api.html>`_
-------------------------------

Detailed API documentation.

`Examples <auto_examples/index.html>`_
--------------------------------------

Further examples that complement the `User Guide <user_guide.html>`_.
