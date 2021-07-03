"""
Example for Importance Transform Sampling

Importance sampling is not actually a method to create samples, it is used to estimate statistical momentums of a
target distribution.
"""

# Import
import numpy as np
import math
import random
from matplotlib import pyplot as plt
from IPython.display import clear_output

def f(x):
    '''
    Target Distribution
    :param x:
    :return:
    '''
    return (math.e**(-1*x))/(1+(x-1)**2)


# Plot the function
xs = [float(i/50) for i in range(int(50*math.pi*2))]
ys = [f(x) for x in xs]
plt.plot(xs,ys)
plt.title("f(x)");
plt.show()


# This is the template of our weight function g(x)
def g(x, A, lamda):
    '''
    Weight function
    :param x:
    :param A:
    :param lamda:
    :return:
    '''
    return A*math.pow(math.e, -1*lamda*x)


xs = [float(i/50) for i in range(int(50*math.pi))]
fs = [f(x) for x in xs]
gs = [g(x, A=1.4, lamda=1.4) for x in xs]
plt.plot(xs, fs)
plt.plot(xs, gs)
plt.title("f(x) and g(x)");
plt.show()


def g_inverse_of_r(r, lamda):
    return (-1 * math.log(float(r)))/lamda


def importance_sampling_variance(lamda, num_samples):
    """
    This function calculates the variance if a Monte Carlo
    using importance sampling.
    Args:
    - lamda (float) : lamdba value of g(x) being tested
    Return:
    - Variance
    """
    A = lamda
    int_max = 5

    # get sum of squares
    running_total = 0
    for i in range(num_samples):
        x = random.uniform(0, int_max)
        running_total += (f(x) / g(x, A, lamda)) ** 2

    sum_of_sqs = running_total / num_samples

    # get squared average
    running_total = 0
    for i in range(num_samples):
        x = random.uniform(0, int_max)
        running_total += f(x) / g(x, A, lamda)
    sq_ave = (running_total / num_samples) ** 2

    return sum_of_sqs - sq_ave


# get variance as a function of lambda by testing many
# different lambdas

test_lamdas = [i * 0.05 for i in range(1, 61)]
variances = []

for i, lamda in enumerate(test_lamdas):
    print(f"lambda {i + 1}/{len(test_lamdas)}: {lamda}")
    A = lamda
    variances.append(importance_sampling_variance(lamda, 10000))
    clear_output(wait=True)

optimal_lamda = test_lamdas[np.argmin(np.asarray(variances))]
IS_variance = variances[np.argmin(np.asarray(variances))]

print(f"Optimal Lambda: {optimal_lamda}")
print(f"Optimal Variance: {IS_variance}")
print((IS_variance / 10000) ** 0.5)

plt.plot(test_lamdas[5:40], variances[5:40])
plt.title("Variance of MC at Different Lambda Values");
plt.show()


def importance_sampling(lamda, num_samples):
    A = lamda

    running_total = 0
    for i in range(num_samples):
        r = random.uniform(0, 1)
        running_total += f(g_inverse_of_r(r, lamda=lamda)) / g(g_inverse_of_r(r, lamda=lamda), A, lamda)
    approximation = float(running_total / num_samples)
    return approximation

# Run simulation
num_samples = 10000
approx = importance_sampling(optimal_lamda, num_samples)
variance = importance_sampling_variance(optimal_lamda, num_samples)
error = (variance/num_samples)**0.5

# display results
print(f"Importance Sampling Approximation: {approx}")
print(f"Variance: {variance}")
print(f"Error: {error}")

__source__ = 'https://towardsdatascience.com/monte-carlo-simulations-with-python-part-1-f5627b7d60b0'