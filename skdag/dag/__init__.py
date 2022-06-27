"""
The :mod:`skdag.dag` module implements utilities to build a composite
estimator, as a directed acyclic graph (DAG). A DAG may have one or more inputs (roots)
and also one or more outputs (leaves), but may not contain any cyclic processing paths.
"""
# Author: big-o (github)
# License: BSD

from skdag.dag._dag import *
from skdag.dag._builder import *
from skdag.dag._render import *
