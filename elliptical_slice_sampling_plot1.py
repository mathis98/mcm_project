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

# TODO: Add plotting
# TODO: Adjust style to Erik

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
        self.iterator = 0
        #sns.set()
        self.c_target = (0.1215686, 0.4666667, 0.70588235)
        self.c_likeli = (0.8392157, 0.1529411, 0.1568627)
        self.c_norm = (1.0, 0.498039, 0.05490196)
        self.c_fp = 'black'
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
        w_range = [-100, 100]
        self.w_vector = np.linspace(w_range[0], w_range[1], 2)

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
                if x_proposal > x_prior:
                    self.w_vector[1] = x_proposal
                elif x_proposal < x_prior:
                    self.w_vector[0] = x_proposal
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
        distribution_target = np.zeros(len(x_vector))
        distribution_lkh = np.zeros(len(x_vector))
        distribution_pool = np.zeros(len(x_vector))
        for idx, val in enumerate(x_vector):
            distribution_target[idx] = sampler.p(val, [mu_total], [sigma_total])
            distribution_lkh[idx] = sampler.p(val, [self.mu_lkh], [self.sigma_lkh])
            distribution_pool[idx] = sampler.p(val, [self.mean], [self.covariance])

        # Set up Samples
        samples = np.zeros([samples_n, 2])
        samples[samples == 0] = -10
        x_prior = np.random.multivariate_normal(mean=self.mean, cov=self.covariance)

        if plot:
            plt.ion()
            figure, ax = plt.subplots(figsize=(10, 6))
            line_distribution_target = ax.plot(x_vector, distribution_target, '-', c=self.c_target, label='target distribution')
            line_distribution_pool = ax.plot(x_vector, distribution_pool, '-', c=self.c_norm, label='$\mathcal{N}(0, \Sigma)$')
            line_distribution_lkh = ax.plot(x_vector, distribution_lkh, '-', c=self.c_likeli, label='likelihood')
            marker_samples, = ax.plot(samples[:, 0], samples[:, 1], '|', c=self.c_fp, markersize=12, alpha=0.6, label='samples')
            marker_x_prior, = ax.plot(0, 0, 'X', markersize=12, c=self.c_target, label='current state')
            line_y_vector, = ax.plot([0] * len(np.linspace(0, 0, 2)), np.linspace(0, 0, 2), '--', c=self.c_likeli)
            marker_y_value, = ax.plot(0, 0, '.', markersize=12, c=self.c_likeli)
            line_w, = ax.plot(np.linspace(0, 1, 2), [0] * len(np.linspace(0, 1, 2)), '--', c=self.c_likeli)
            plt.title("elliptical slice sampling")
            plt.legend()
            plt.grid(True)
            plt.xlim(x_range)
            plt.ylim([-0.1, 0.5])
            plt.xlabel("sample")
            plt.ylabel("value")
            self.iterator +=1
            plt.savefig('ess_' + str(self.iterator) + '.png')

        for i in range(1, samples_n):
            sample = sampler.sample(x_prior, plot=True, plot_timer=plot_timer)
            samples[i - 1, 0] = sample[0]
            #samples[i - 1, 1] = sample[1]
            samples[i - 1, 1] = -0.02

            # Update Plot
            if plot:
                # Updating data values
                marker_samples.set_xdata(samples[:, 0])
                marker_samples.set_ydata(samples[:, 1])
                figure.canvas.flush_events()
                time.sleep(plot_timer)
                self.iterator += 1
                plt.savefig('ess_' + str(self.iterator) + '.png')
                marker_x_prior.set_xdata(x_prior)
                figure.canvas.flush_events()
                time.sleep(plot_timer)
                self.iterator += 1
                plt.savefig('ess_' + str(self.iterator) + '.png')
                line_y_vector.set_xdata([x_prior, x_prior])
                line_y_vector.set_ydata([0, sampler.p(x_prior, [self.mu_lkh], [self.sigma_lkh])])
                figure.canvas.flush_events()
                time.sleep(plot_timer)
                self.iterator += 1
                plt.savefig('ess_' + str(self.iterator) + '.png')
                marker_y_value.set_xdata(x_prior)
                marker_y_value.set_ydata(sample[1])
                figure.canvas.flush_events()
                time.sleep(plot_timer)
                self.iterator += 1
                plt.savefig('ess_' + str(self.iterator) + '.png')
                line_w.set_xdata(self.w_vector)
                line_w.set_ydata([sample[1]] * len(self.w_vector))
                figure.canvas.flush_events()
                time.sleep(plot_timer)
                self.iterator += 1
                plt.savefig('ess_' + str(self.iterator) + '.png')

                # drawing updated values
                figure.canvas.draw()

                # This will run the GUI event
                # loop until all UI events
                # currently waiting have been processed
                figure.canvas.flush_events()
                time.sleep(plot_timer)

            x_prior = sample[0]
        return samples

# Run Sampler
begin_time = datetime.datetime.now()

samples_n = 1000
mu = 0.0
sigma = 1.0
x_range = [-5.0, 15]
mu_lkh = 4.0
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