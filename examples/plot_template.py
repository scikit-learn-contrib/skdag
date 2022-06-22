"""
===========================
Plotting Template Estimator
===========================

An example plot of :class:`skdag.template.TemplateEstimator`
"""
import numpy as np
from matplotlib import pyplot as plt
from skdag import TemplateEstimator

X = np.arange(100).reshape(100, 1)
y = np.zeros((100,))
estimator = TemplateEstimator()
estimator.fit(X, y)
plt.plot(estimator.predict(X))
plt.show()
