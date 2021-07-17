"""
Script for Elliptical Slice Sampling
"""

import faulthandler
import scipy.stats as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
import statistics
import math
import datetime
import time
import logging

class EllipticalSliceSampler:
    def __init__(self, mean, covariance, mu_lkh, sigma_lkh, x_range, seed):
        logging.basicConfig(level=logging.INFO)
        self.mean = mean
        self.covariance = covariance
        self.mu_lkh = mu_lkh
        self.sigma_lkh = sigma_lkh
        self.x_range = x_range
        self.seed = seed
        self.accepted = 0
        self.rejected = 0
        self.rejected_internal = 0
        np.random.seed(self.seed)
        #sns.set()
        self.c_target = (0.1215686, 0.4666667, 0.70588235)
        np.random.seed(seed)
        faulthandler.enable()

    def log_likelihood_func(self, x):
        """
        Defines the likelihood function.
        :param x:
        :return:
        """
        return norm.logpdf(x, self.mu_lkh, self.sigma_lkh)

    def p(self, x, mu, sigma):
        """
        Defines a 1D Target Distribution.
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

    def sample(self, x_prior, plot=True, plot_timer = 0.0):
        """
        Creates a single Sample.
        :param x_prior:
        :return:
        """
        nu = np.random.multivariate_normal(np.zeros(self.mean.shape), self.covariance)
        y_log = sampler.log_likelihood_func(x_prior) + np.log(np.random.uniform())
        y_proposal = np.exp(y_log)
        theta = np.random.uniform(0., 2.*np.pi)
        theta_min, theta_max = theta - 2.*np.pi, theta

        # Sample Value
        sample_found = False
        while not sample_found:
            x_proposal = (x_prior - self.mean)*np.cos(theta) + nu*np.sin(theta) + self.mean
            x_proposal_log = sampler.log_likelihood_func(x_proposal)
            if x_proposal_log > y_log:
                logging.debug('Accepted')
                sample_found = True
                sample = [x_proposal, y_proposal]
                self.accepted += 1
            else:
                logging.debug('Rejected Internal')
                self.rejected_internal += 1
                if theta < 0.:
                    theta_min = theta
                else:
                    theta_max = theta
                theta = np.random.uniform(theta_min, theta_max)
        return sample

    def sample_1d(self, samples_n, plot=False, plot_timer=0.0):
        """
        Creates n samples in 1d
        :return: sample
        """

        # Create Target Distribution
        x_vector = np.linspace(self.x_range[0], self.x_range[1], 1000)
        mu_total = ((self.covariance ** -2) * self.mean + (self.sigma_lkh ** -2) * self.mu_lkh) / (self.covariance ** -2 + self.sigma_lkh ** -2)
        sigma_total = np.sqrt((self.covariance ** 2 * self.sigma_lkh ** 2) / (self.covariance ** 2 + self.sigma_lkh ** 2))
        target_distribution = np.zeros(len(x_vector))
        for idx, val in enumerate(x_vector):
            target_distribution[idx] = sampler.p(val, [mu_total], [sigma_total])

        # Set up Samples
        samples = np.zeros([samples_n, 2])
        samples[samples == 0] = -10
        x_prior = np.random.multivariate_normal(mean=self.mean, cov=self.covariance)

        # Set up Plotting
        if plot:
            plt.ion()
            figure, ax = plt.subplots(1, 2, figsize=(15, 7))

        for i in range(1, samples_n):
            sample = sampler.sample(x_prior, plot=True, plot_timer=plot_timer)
            samples[i-1, 0] = sample[0]
            samples[i-1, 1] = sample[1]

            # Update Plot
            if plot:
                samples_accepted = samples[~np.all(samples == -10, axis=1)]
                # Figure 1
                ax[0].cla()
                ax[0].plot(target_distribution, x_vector, '-', markersize=12, c=self.c_target)
                #ax[0].plot(y_vector, samples, 'xk', linewidth=2, markersize=10)
                ax[0].hist(samples[:, 0], bins=30, color='g', density=True, alpha=0.6, orientation="horizontal")
                ax[0].set_title("target distribution & histogram", fontsize=16)
                ax[0].text(0.2, 8.5, 'iteration: {} \naccepted: {} \nrejected: {}'.format(i, self.accepted, self.rejected), fontsize=16)
                ax[0].set_xlim([-0.1, 0.5])
                ax[0].invert_xaxis()
                ax[0].set_ylim(x_range)
                ax[0].set_xlabel('value')
                ax[0].set_ylabel('sample')
                ax[0].grid()
                ax[0].legend(['target Distribution', 'histogram'])

                increment_vector = np.linspace(0, len(samples_accepted), num=len(samples_accepted))
                # Figure 2
                ax[1].cla()
                ax[1].plot(increment_vector, samples_accepted[:, 0], '-', markersize=12, c=self.c_target)
                ax[1].set_title("samples", fontsize=16)
                ax[1].set_ylim(x_range)
                ax[1].set_xlim([0, len(samples_accepted)])
                ax[1].set_xlabel("iteration")
                ax[1].set_ylabel("sample")
                ax[1].grid()
                ax[1].legend(['samples'])

                time.sleep(plot_timer)
                figure.canvas.flush_events()
                figure.canvas.draw()
            x_prior = sample[0]
        return samples

# Run Sampler
begin_time = datetime.datetime.now()

samples_n = 1000
mu = 5.0
sigma = 1.0
x_range = [0.5, 10]
mu_lkh = 1.0
sigma_lkh = 2.0
seed = 0
plot_timer = 0.0

sampler = EllipticalSliceSampler(np.array([mu]), np.diag(np.array([sigma**2, ])), mu_lkh, sigma_lkh, x_range, seed)
samples = sampler.sample_1d(samples_n=samples_n, plot=True, plot_timer=plot_timer)

# Plot End Results
mu_total = ((sigma**-2)*mu + (sigma_lkh**-2)*mu_lkh) / (sigma**-2 + sigma_lkh**-2)
sigma_total = np.sqrt((sigma**2 * sigma_lkh**2) / (sigma**2 + sigma_lkh**2))

x = np.linspace(x_range[0], x_range[1], 1000)
target_distribution = np.zeros(len(x))
for idx, val in enumerate(x):
    target_distribution[idx] = sampler.p(x[idx], mu_total, sigma_total)

target_distribution_norm = [float(i)/sum(target_distribution) for i in target_distribution]

plt.figure(figsize=(17, 6))
plt.hist(samples[:, 0], bins=30, color='b', density=True, alpha=0.6)
plt.plot(x, target_distribution, '-r', linewidth=2)
plt.title("Elliptical Slice Sampling", fontsize=16)
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