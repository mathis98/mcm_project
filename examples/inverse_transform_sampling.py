"""
Example for Inverse Transform Sampling
"""

# Imports
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import uniform, expon
import pandas as pd


# -- Continuous Variables -- #
def exponential_inverse_trans(n=1, mean=1):
    '''
    Generate exponential distributed random variables given the mean and number of random variables.
    CFD = 1-e^(-x/mean)
    :param n:
    :param mean:
    :return:
    '''

    u = uniform.rvs(size=n)
    x = -mean * np.log(1 - u)  # CFD^-1
    actual = expon.rvs(size=n, scale=mean)

    plt.figure(figsize=(12, 9))
    plt.hist(x, bins=50, alpha=0.5, label="Generated r.v.")
    plt.hist(actual, bins=50, alpha=0.5, label="Actual r.v.")
    plt.title("Generated vs Actual %i Exponential Random Variables" % n)
    plt.legend()
    plt.show()
    return x


exponential_inverse_trans(1000, 1)


# -- Discrete Variables -- #
def discrete_inverse_trans(prob_vec):
    '''
    Generate arbitrary discrete distributed random variables given the probability vector.
    :param prob_vec:
    :return:
    '''

    u = uniform.rvs(size=1)
    if u <= prob_vec[0]:
        return 1
    else:
        for i in range(1, len(prob_vec) + 1):
            if sum(prob_vec[0:i]) < u < sum(prob_vec[0:i + 1]):
                return i + 1


def discrete_samples(prob_vec, n=1):
    '''
    Create n Samples given the probability vector.
    :param prob_vec:
    :param n:
    :return:
    '''
    sample = []
    for i in range(0, n):
        sample.append(discrete_inverse_trans(prob_vec))
    return np.array(sample)


def discrete_simulate(prob_vec, numbers, n=1):
    '''
    Simulate Inverse Transform Sampling for n samples, given the probability vector.
    :param prob_vec:
    :param numbers:
    :param n:
    :return:
    '''
    sample_disc = discrete_samples(prob_vec, n)
    unique, counts = np.unique(sample_disc, return_counts=True)

    fig = plt.figure() # TODO: Axis Missing, Plot with true values as comparison
    ax = fig.add_axes([0, 0, 1, 1])
    prob = counts / n
    ax.bar(numbers, prob)
    plt.title("Simulation of Generating %i Discrete Random Variables" % n)
    plt.show()

    data = {'X': unique, 'Number of samples': counts, 'Empirical Probability': prob, 'Actual Probability': prob_vec}
    df = pd.DataFrame(data=data)
    return df

prob_vec = np.array([0.1, 0.3, 0.5, 0.05, 0.05])
numbers = np.array([1, 2, 3, 4, 5])
dis_example1 = discrete_simulate(prob_vec, numbers, n = 1000)

__author__ = 'Raden Aurelius Andhika Viadinugroho'
__source__ = 'https://towardsdatascience.com/generate-random-variable-using-inverse-transform-method-in-python-8e5392f170a3'
