.. project-template documentation master file, created by
   sphinx-quickstart on Mon Jan 18 14:44:12 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

skdag - scikit-learn workflow management
============================================

scikit-dag (``skdag``) is an open-sourced, MIT-licenced library that provides advanced
workflow management to any machine learning operations that follow
:mod:`sklearn<scikit-learn>` conventions. It does this by introducing Directed Acyclic
Graphs (:class:`skdag.dag.DAG`) as a drop-in replacement for traditional scikit-learn
:mod:`sklearn.pipeline.Pipeline<pipelines>`.

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