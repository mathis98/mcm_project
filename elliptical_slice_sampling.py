"""
Script for Elliptical Slice Sampling
"""

# TODO: Move Elliptical Slice Sampling to Our Scheme
# TODO: Add Hastings to Metropolis Hastings
# TODO: Internal Rejection Rate
# TODO: Computation Time
# TODO: Add 2d Case

# Imports
import numpy as np
import scipy.stats as st
from scipy.stats import norm
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import time
import statistics
import math


class EllipticalSliceSampler:
    def __init__(self, mu, sigma, x_range, seed):
        """
        Slice Sampler
        :param mu:
        :param sigma:
        :param x_range:
        :param w_length:
        :param seed:
        """
        logging.basicConfig(level=logging.INFO)
        self.mu = mu
        self.sigma = sigma
        self.x_range = x_range
        np.random.seed(seed)
        sns.set()

    def p(self, x, mu, sigma):
        """
        Defines a 1D Target Distribution
        :param x: x value
        :param mu: List of Expected Values
        :param sigma: List of Standard Deviations
        :return:
        """
        p = np.zeros(len(mu))
        for idx, mu_val in enumerate(mu):
            sigma_val = sigma[idx]
            p[idx] = st.norm.pdf(x, loc=mu_val, scale=sigma_val)
        return sum(p)

    def log_likelihood_func(self, x, mu, sigma):
        """
        Log Likelihood Function
        :param x:
        :param mu:
        :param sigma:
        :return:
        """
        return norm.logpdf(x, mu, sigma)

    def sample(self, x_prior):
        """
        Creates a single Sample. x_value is the prior x
        :param x_prior:
        :return:
        """
        # Choose Ellipse
        #nu = np.random.normal(0, 1, 1) # TODO: Which to use?
        nu = np.random.multivariate_normal(np.zeros(self.mu.shape), self.covariance)

        # Log-likelihood threshold
        u = np.random.uniform()
        y_log = sampler.log_likelihood_func(x_prior, 0, sigma=1) + np.log(u) # TODO: Fix first Term
        y_value = 0.0

        # Initial Proposal for theta
        theta = np.random.uniform(0., 2. * np.pi)
        theta_min, theta_max = theta - 2. * np.pi, theta

        # Sample value
        sample = [0, 0]
        sample_found = False
        while not sample_found:
            # Generate Proposal Point on Ellipse
            x_proposal = (x_prior - self.mu[0]) * np.cos(theta) + nu * np.sin(theta) + self.mu[0] #TODO: self.mu[0] correct?
            x_proposal_log = sampler.log_likelihood_func(x_proposal, 0, sigma=1) # TODO: Fix
            # Accepts
            if x_proposal_log > y_log:
                sample_found = True
                sample[0] = x_proposal
                sample[1] = y_value
                return sample
            # Reject
            else:
                # Shrink the Bracket, till a sample is accepted
                if theta < 0.:
                    theta_min = theta
                else:
                    theta_max = theta
                theta = np.random.uniform(theta_min, theta_max)

        return sample


    def samples(self, samples_n, plot = False):
        """
        Creates n samples in 1d
        :return: sample
        """

        # Create Target Distribution
        x_vector = np.linspace(self.x_range[0], self.x_range[1], 1000)
        target_distribution = np.zeros(len(x_vector))
        for idx, val in enumerate(x_vector):
            target_distribution[idx] = sampler.p(val, self.mu, self.sigma)

        # Create Samples
        samples = np.empty([samples_n, 2])

        if plot:
            plt.ion()
            figure, ax = plt.subplots(figsize=(10, 8))
            line_distribution, = ax.plot(x_vector, target_distribution, '-', linewidth=2, markersize=1)
            marker_x_value, = ax.plot(0, 0, 'xr', linewidth=2, markersize=10)
            line_y_vector, = ax.plot([0] * len(np.linspace(0, 1, 2)), np.linspace(0, 1, 2), '-g', linewidth=2, markersize=1)
            marker_y_value, = ax.plot(0, 0, 'xy', linewidth=1, markersize=10)
            line_w, = ax.plot(np.linspace(0, 1, 2), [0] * len(np.linspace(0, 1, 2)), '-m', linewidth=2, markersize=10)
            marker_samples, = ax.plot(samples[:, 0], samples[:, 1], 'xb', linewidth=2, markersize=10)
            plt.title("Slice Sampling", fontsize=16)
            plt.xlim(x_range)
            plt.ylim([-0.1, 0.75])
            plt.xlabel("X")
            plt.ylabel("Y")

        # Create Samples
        x_value = np.random.uniform(self.x_range[0], self.x_range[1])  # Random x (Slice) for first sample
        for i in range(samples_n):
            sample = sampler.sample(x_value)
            samples[i, 0] = sample[0]
            samples[i, 1] = sample[1]

            if plot:
                # updating data values
                timer = 0.001
                y_value = sample[1]
                line_distribution.set_xdata(x_vector)
                line_distribution.set_ydata(target_distribution)
                figure.canvas.flush_events()
                time.sleep(timer)
                marker_x_value.set_xdata(x_value)
                marker_x_value.set_ydata(0)
                figure.canvas.flush_events()
                time.sleep(timer)
                line_y_vector.set_xdata([x_value] * len(self.y_vector))
                line_y_vector.set_ydata(self.y_vector)
                figure.canvas.flush_events()
                time.sleep(timer)
                marker_y_value.set_xdata(x_value)
                marker_y_value.set_ydata(y_value)
                figure.canvas.flush_events()
                time.sleep(timer)
                line_w.set_xdata(self.w_vector)
                line_w.set_ydata([y_value] * len(self.w_vector))
                figure.canvas.flush_events()
                time.sleep(timer)
                marker_samples.set_xdata(samples[:, 0])
                marker_samples.set_ydata(samples[:, 1])
                figure.canvas.flush_events()
                time.sleep(timer)

                # drawing updated values
                figure.canvas.draw()

                # This will run the GUI event loop until all UI events currently waiting have been processed
                figure.canvas.flush_events()
                time.sleep(timer)

            x_value = sample[0]

        return samples


# Run Sampler
samples_n = 1000
mu = [15, 20]
sigma = [1, 3]
x_range = [5, 35]
w_length = 0.5
seed = 0

mu_total = ((sigma[0]**-2)*mu[0] + (sigma[1]**-2)*mu[1]) / (sigma[0]**-2 + sigma[1]**-2)
sigma_total = np.sqrt((sigma[0]**2 * sigma[1]**2) / (sigma[0]**2 + sigma[1]**2))

sampler = EllipticalSliceSampler(mu, sigma, x_range, seed)
samples = sampler.samples(samples_n, plot=False)

# Plot End Result
x = np.linspace(x_range[0], x_range[1], 1000)
target_distribution = np.zeros(len(x))
for idx, val in enumerate(x):
    target_distribution[idx] = sampler.p(x[idx], mu, sigma)

target_distribution_norm = [float(i)/sum(target_distribution) for i in target_distribution]

plt.figure(figsize=(17, 6))
plt.hist(samples[:, 0], bins=30, color='b', density=True, alpha=0.6)
plt.plot(x, target_distribution, '-r', linewidth=2)
plt.title("Slice Sampling", fontsize=16)
plt.xlim(x_range)
plt.ylim([-0.1, 0.75])
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

# Momentums
samples_mean = statistics.mean(samples[:, 0])
samples_std = statistics.stdev(samples[:, 0])
target_mean = sum(np.multiply(x, target_distribution_norm))
target_std = math.sqrt(sum(np.multiply(target_distribution_norm, (x-target_mean)**2)))

print('Mean of Samples: {}'.format(samples_mean))
print('Mean of Target: {}'.format(target_mean))
print('Standard Deviation of Samples: {}'.format(samples_std))
print('Standard Deviation of Target: {}'.format(target_std))
print('Standard Error: {}'.format((samples_std/samples_n)**0.5))

__source__ = ''

'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

class EllipticalSliceSampler:
    def __init__(self, mean, covariance, log_likelihood_func):
        self.mean = mean
        self.covariance = covariance
        self.function_likelihood_log = log_likelihood_func

    def sample(self, f):
        """
        Creates a single sample.
        :param f:
        :return: fp
        """

        nu = np.random.multivariate_normal(np.zeros(self.mean.shape), self.covariance) # covariance from target distribution

        # Log-likelihood threshold
        u = np.random.uniform()
        log_y = self.function_likelihood_log(f) + np.log(u)

        # Initial Proposal for theta
        theta = np.random.uniform(0., 2.*np.pi)
        theta_min, theta_max = theta - 2.*np.pi, theta

        # Shrink the Bracket, till a sample is accepted
        while True:
            # Generate Point on Ellipse
            fp = (f - self.mean)*np.cos(theta) + nu*np.sin(theta) + self.mean
            log_fp = self.function_likelihood_log(fp)
            # Accept
            if log_fp > log_y:
                return fp
            # Reject
            else:
                # Create new theta
                if theta < 0.:
                    theta_min = theta
                else:
                    theta_max = theta
                theta = np.random.uniform(theta_min, theta_max)


    def sample_1d(self, samples_n, burnin=1000):
        """
        Creates n samples.
        :param samples_n:
        :param burnin:
        :return:
        """
        total_samples_n = samples_n + burnin
        samples = np.zeros((total_samples_n, self.covariance.shape[0]))
        samples[0] = np.random.multivariate_normal(
            mean=self.mean, cov=self.covariance)

        for i in range(1, total_samples_n):
            samples[i] = self.sample(samples[i-1])
        return samples[burnin:]

np.random.seed(0)

mu_1 = 5.
mu_2 = 1.
sigma_1, sigma_2 = 1., 2.
mu = ((sigma_1**-2)*mu_1 + (sigma_2**-2)*mu_2) / (sigma_1**-2 + sigma_2**-2)
sigma = np.sqrt((sigma_1**2 * sigma_2**2) / (sigma_1**2 + sigma_2**2))

def log_likelihood_func(f):
    return norm.logpdf(f, mu_2, sigma_2)

samples_n = 10000
burnin = 1000
sampler = EllipticalSliceSampler(np.array([mu_1]), np.diag(np.array([sigma_1**2, ])), log_likelihood_func)

samples = sampler.sample_1d(samples_n=samples_n, burnin=burnin)

r = np.linspace(0., 8., num=100)
plt.figure(figsize=(17, 6))
#plt.hist(samples, bins=30, normed = true)
plt.hist(samples, bins=30)
plt.plot(r, norm.pdf(r, mu, sigma))
plt.grid()
plt.show()

__author__ = 'Viking Penguin'
__source__ = 'https://www.youtube.com/watch?v=HfzyuD9_gmk&t=448s'
'''