"""
Example for Metropolis Hastings
"""

# Imports
import numpy as np
import scipy.stats as st
import seaborn as sns
from matplotlib import pyplot as plt

def circle(x, y):
    return (x-1)**2 + (y-2)**2 - 3**2


def pgauss(x, y):
    return st.multivariate_normal.pdf([x, y], mean=mus, cov=sigmas)


def metropolis_hastings(p, iter=1000):
    x, y = 0., 0.
    samples = np.zeros((iter, 2))

    for i in range(iter):
        x_star, y_star = np.array([x, y]) + np.random.normal(size=2)
        if np.random.rand() < p(x_star, y_star) / p(x, y):
            x, y = x_star, y_star
        samples[i] = np.array([x, y])

    return samples

mus = np.array([5, 5])
sigmas = np.array([[1, .9], [.9, 1]])

samples = metropolis_hastings(circle, iter=10000)
sns.jointplot(samples[:, 0], samples[:, 1])
plt.show()

samples = metropolis_hastings(pgauss, iter=10000)
sns.jointplot(samples[:, 0], samples[:, 1])
plt.show()

__source__ = 'https://wiseodd.github.io/techblog/2015/10/17/metropolis-hastings/'