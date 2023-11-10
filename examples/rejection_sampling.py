"""
Example for Rejection Sampling
"""

# Imports
import numpy as np
import scipy.stats as st
import seaborn as sns
import matplotlib.pyplot as plt


# -- Continuous Variables -- #
sns.set()


def p(x):
    '''
    Target Distribution
    :param x:
    :return:
    '''
    return st.norm.pdf(x, loc=30, scale=10) + st.norm.pdf(x, loc=80, scale=20)


def q(x):
    '''
    Proposal Distribution
    :param x:
    :return:
    '''
    return st.norm.pdf(x, loc=50, scale=30)


def rejection_sampling(iter = 1000):
    '''
    Rejection Sampler
    :param iter:
    :return:
    '''
    samples = []

    for i in range(iter):
        z = np.random.normal(50, 30)
        u = np.random.uniform(0, scale_factor * q(z))

        if u/p(z) <= 1:
            samples.append(z)

    return np.array(samples)


samples_number = 1000000
x = np.arange(-50, 151)
scale_factor = max(p(x) / q(x))
samples = rejection_sampling(iter=100000)

plt.plot(x, p(x))
plt.plot(x, scale_factor * q(x))
plt.title('Target and Proposal Distributions')
plt.show()

sns.displot(samples)
plt.title('Samples from Rejection Sampler')
plt.show()

#TODO: Numbers of rejected samples, Efficiency (Ratio of accepted samples), Computation Time, Error, Standard Deviation.


__author__ = ''
__source__ = 'https://wiseodd.github.io/techblog/2015/10/21/rejection-sampling/'

