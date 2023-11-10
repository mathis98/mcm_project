"""
Example for Naive Monte Carlo Method. (Also known as Crude Monte Carlo)

Introduction to Importance Sampling
"""

# Import
import numpy as np
import math
import random
import math
from matplotlib import pyplot as plt
from IPython.display import clear_output

def f(x):
    '''
    Target Distribution
    :param x:
    :return:
    '''
    return (math.e**(-1*x))/(1+(x-1)**2)


def crude_monte_carlo(num_samples=1):
    """
    This function performs the Crude Monte Carlo for our specific function f(x) on the range x=0 to x=5.
    Notice that this bound is sufficient because f(x) approaches 0 at around PI.
    Args:
    - num_samples (float) : number of samples
    Return:
    - Crude Monte Carlo estimation (float)

    """
    lower_bound = 0
    upper_bound = 5

    sum_of_samples = 0
    for i in range(num_samples):
        x = random.uniform(lower_bound, upper_bound)
        sum_of_samples += f(x)
    return (upper_bound - lower_bound) * float(sum_of_samples / num_samples)


def naive_mc_variance(num_samples):
    """
    This function returns the variance fo the Crude Monte Carlo.
    Note that the inputted number of samples does not necessarily need to correspond to number of samples
    used in the Monte Carlo Simulation.
    Args:
    - num_samples (int)
    Return:
    - Variance for Crude Monte Carlo approximation of f(x) (float)
    """
    int_max = 5  # this is the max of our integration range

    # get the average of squares
    running_total = 0
    for i in range(num_samples):
        x = random.uniform(0, int_max)
        running_total += f(x) ** 2
    sum_of_sqs = running_total * int_max / num_samples

    # get square of average
    running_total = 0
    for i in range(num_samples):
        x = random.uniform(0, int_max)
        running_total = f(x)
    sq_ave = (int_max * running_total / num_samples) ** 2
    return sum_of_sqs - sq_ave


# Now we will run a Crude Monte Carlo simulation with 10000 samples
# We will also calculate the variance with 10000 samples and the error
samples_n = 10000
crude_estimation = crude_monte_carlo(samples_n)
variance = naive_mc_variance(samples_n)
error = math.sqrt(variance/samples_n)

# display results
print(f"Monte Carlo Approximation of f(x): {crude_estimation}")
print(f"Variance of Approximation: {variance}")
print(f"Error in Approximation: {error}")

__source__ = 'https://towardsdatascience.com/monte-carlo-simulations-with-python-part-1-f5627b7d60b0'