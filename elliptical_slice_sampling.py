"""
Script for Elliptical Slice Sampling
"""

# TODO: Add Functionality for Gaussian Mixture

import faulthandler
import scipy.stats as st
import numpy as np
from scipy.stats import norm
from numpy.core.fromnumeric import mean
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
import statistics
import math
import datetime
import time

class EllipticalSliceSampler:
    def __init__(self, mean, covariance, mu_lkh, sigma_lkh, x_range, seed):
        self.mean = mean
        self.covariance = covariance
        self.mu_lkh = mu_lkh
        self.sigma_lkh = sigma_lkh
        self.x_range = x_range
        self.seed = seed
        self.accepted = 0
        self.rejected = 0
        np.random.seed(self.seed)
        sns.set()
        faulthandler.enable()

    def log_likelihood_func(self, x):
        return norm.logpdf(x, self.mu_lkh, self.sigma_lkh)

    def p(self, x, mu, sigma):
        """
        Defines a 1D Target Distribution
        :param x: x value
        :param mu: List of Means
        :param sigma: List of Standard Deviations
        :return:
        """
        p = np.zeros(len([mu]))
        for idx, mu_val in enumerate([mu]):
            sigma_val = [sigma][idx]
            p[idx] = st.norm.pdf(x, loc=mu_val, scale=sigma_val) # Scale is Standard Deviation
        return sum(p)

    def sample(self, x_prior):
        """
        Creates a single Sample. x_value is the prior x
        :param x_prior:
        :return:
        """
        nu = np.random.multivariate_normal(np.zeros(self.mean.shape), self.covariance)
        y_log = sampler.log_likelihood_func(x_prior) + np.log(np.random.uniform())
        theta = np.random.uniform(0., 2.*np.pi)
        theta_min, theta_max = theta - 2.*np.pi, theta

        sample = [0, 0]
        # Sample Value
        sample_found = False
        while not sample_found:
            x_proposal = (x_prior - self.mean)*np.cos(theta) + nu*np.sin(theta) + self.mean
            x_proposal_log = sampler.log_likelihood_func(x_proposal)
            if x_proposal_log > y_log:
                sample_found = True
                self.accepted += 1
                return x_proposal
            else:
                if theta < 0.:
                    theta_min = theta
                else:
                    theta_max = theta
                theta = np.random.uniform(theta_min, theta_max)

    def sample_1d(self, samples_n, plot = False):
        """
        Creates n samples in 1d
        :return: sample
        """

        # Create Target Distribution
        x_vector = np.linspace(self.x_range[0], self.x_range[1], 1000)
        [mu_total] = ((self.covariance ** -2) * self.mean + (self.sigma_lkh ** -2) * self.mu_lkh) / (self.covariance ** -2 + self.sigma_lkh ** -2)
        [sigma_total] = np.sqrt((self.covariance ** 2 * self.sigma_lkh ** 2) / (self.covariance ** 2 + self.sigma_lkh ** 2))
        target_distribution = np.zeros(len(x_vector))
        for idx, val in enumerate(x_vector):
            target_distribution[idx] = sampler.p(val, [mu_total], [sigma_total])

        samples = np.zeros((samples_n, self.covariance.shape[0]))
        samples[samples == 0] = -10
        y_vector = np.zeros((samples_n, self.covariance.shape[0]))
        samples[0] = np.random.multivariate_normal(mean=self.mean, cov=self.covariance)

        # Set up Plotting
        if plot:
            plt.ion()
            figure, ax = plt.subplots(figsize=(10, 8))

        for i in range(1, samples_n):
            samples[i] = self.sample(samples[i-1])

            # Update Plot
            if plot:
                # updating data values
                timer = 0.0001
                ax.cla()
                line_distribution, = ax.plot(x_vector, target_distribution, '-', linewidth=2, markersize=1)
                #marker_samples, = ax.plot(samples, y_vector, 'xb', linewidth=2, markersize=10)
                plt.hist(samples, bins=30, color='b', density=True, alpha=0.6)
                plt.title("Metropolis Hastings Sampling", fontsize=16)
                plt.xlim(x_range)
                plt.ylim([-0.1, 0.75])
                plt.xlabel("X")
                plt.ylabel("Y")
                #time.sleep(timer)
                figure.canvas.flush_events()
                figure.canvas.draw()

        return samples

# Run Sampler
begin_time = datetime.datetime.now()

samples_n = 1000
mu = 5.0
sigma = 1.0
x_range = [-0, 10]
mu_lkh = 1.0
sigma_lkh = 2.0
seed = 0

sampler = EllipticalSliceSampler(np.array([mu]), np.diag(np.array([sigma**2, ])), mu_lkh, sigma_lkh, x_range, seed) #TODO: Add possiblity for mixture of gaussians
samples = sampler.sample_1d(samples_n=samples_n, plot=True)

# Plot End Results
mu_total = ((sigma**-2)*mu + (sigma_lkh**-2)*mu_lkh) / (sigma**-2 + sigma_lkh**-2)
sigma_total = np.sqrt((sigma**2 * sigma_lkh**2) / (sigma**2 + sigma_lkh**2))
#p = norm.pdf(x, mu_total, sigma_total)

x = np.linspace(x_range[0], x_range[1], 1000)
target_distribution = np.zeros(len(x))
for idx, val in enumerate(x):
    target_distribution[idx] = sampler.p(x[idx], mu_total, sigma_total)

target_distribution_norm = [float(i)/sum(target_distribution) for i in target_distribution]

plt.figure(figsize=(17, 6))
plt.hist(samples, bins=30, color='b', density=True, alpha=0.6)
plt.plot(x, target_distribution, '-r', linewidth=2)
#plt.plot(x, p, 'k', linewidth=2)
plt.title("Slice Sampling", fontsize=16)
plt.xlim(x_range)
plt.ylim([-0.1, 0.75])
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

# Momentums
x = np.linspace(x_range[0], x_range[1], 1000)
samples_mean = statistics.mean(samples[:, 0])
samples_std = statistics.stdev(samples[:, 0])
target_mean = sum(np.multiply(x, target_distribution_norm))
target_std = math.sqrt(sum(np.multiply(target_distribution_norm, (x-target_mean)**2)))

print('Mean of Samples: {}'.format(samples_mean))
print('Mean of Target: {}'.format(target_mean))
print('Standard Deviation of Samples: {}'.format(samples_std))
print('Standard Deviation of Target: {}'.format(target_std))
print('Standard Error: {}'.format((samples_std/samples_n)**0.5))
print('Execution Time: {}'.format(datetime.datetime.now() - begin_time))

__source__ = ''