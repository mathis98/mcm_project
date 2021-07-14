"""
Metropolis Hastings Sampling
"""

# TODO: Add Hastings
# TODO: Add 2d Case

import numpy as np
import scipy.stats as st
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import time
import statistics
import math
import datetime


class MetropolisHastingsSampler:
    def __init__(self, mu, sigma, x_range, step_size, seed):
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
        self.step_size = step_size
        self.rejections = 0
        self.acceptances = 0
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

    def sample(self, x_prior, y_prior):
        """
        Creates a single Sample. x_value is the prior x
        :param x_value: float
        :param y_value: float
        :return: array sample
        """
        sample = [0, 0]

        mu_total = ((sigma[0] ** -2) * mu[0] + (sigma[1] ** -2) * mu[1]) / (sigma[0] ** -2 + sigma[1] ** -2)
        sigma_total = np.sqrt((sigma[0] ** 2 * sigma[1] ** 2) / (sigma[0] ** 2 + sigma[1] ** 2))

        # Propose Sample
        x_proposed = np.random.normal(x_prior, sigma_total, 1)  # This is symmetric, so metropolis
        #nu = np.random.normal(x_prior, sigma_total, 1)
        #x_proposed = math.sqrt(1-(self.step_size**2))*x_prior + self.step_size * nu

        y_proposed = 0.0

        # Accept / Reject
        p_prior = sampler.p(x_prior, self.mu, self.sigma)
        p_proposed = sampler.p(x_proposed, self.mu, self.sigma)

        logging.debug('Prior Sample:{}{}'.format(x_prior, y_prior))
        logging.debug('Proposed Sample:{}{}'.format(x_proposed, y_proposed))
        logging.debug(('Probability Prior {}'.format(p_prior)))
        logging.debug(('Probability Proposed {}'.format(p_proposed)))

        g_ab = 1. # TODO: Adjust here for Hastings
        g_ba = 1.
        rf = p_proposed/p_prior
        rg = g_ab/g_ba

        acceptance_ratio = rf * rg
        acceptance_probability = min(1, acceptance_ratio)
        logging.debug('Acceptance Probability: {}'.format(acceptance_probability))


        if np.random.rand() < acceptance_probability:
            accept = True
            self.acceptances += 1
            logging.debug('Sample Accepted: [{} {}]'.format(x_proposed, y_proposed))
            sample[0] = x_proposed
            sample[1] = y_proposed
        else:
            accept = False
            self.rejections += 1
            logging.debug('Sample Rejected')

        return sample, accept


    def sample_1d(self, samples_n, burnin, plot = False):
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

        # Set up Plotting
        if plot:
            plt.ion()
            figure, ax = plt.subplots(figsize=(10, 8))
            line_distribution, = ax.plot(x_vector, target_distribution, '-', linewidth=2, markersize=1)
            marker_x_value, = ax.plot(0, 0, 'xr', linewidth=2, markersize=10)
            marker_y_value, = ax.plot(0, 0, 'xy', linewidth=1, markersize=10)
            marker_samples, = ax.plot(samples[:, 0], samples[:, 1], 'xb', linewidth=2, markersize=10)
            plt.title("Metropolis Hastings Sampling", fontsize=16)
            plt.xlim(x_range)
            plt.ylim([-0.1, 0.75])
            plt.xlabel("X")
            plt.ylabel("Y")

        # Create Samples
        x_value = np.random.uniform(self.x_range[0], self.x_range[1])  # Random x (Slice) for first sample
        y_value = 0.                                                   # y for first sample
        for i in range(samples_n):
            sample, accept = sampler.sample(x_value, y_value)
            samples[i, 0] = sample[0]
            samples[i, 1] = sample[1]

            # Update Plot
            if plot:
                # updating data values
                timer = 0.0001
                y_value = sample[1]
                line_distribution.set_xdata(x_vector)
                line_distribution.set_ydata(target_distribution)
                figure.canvas.flush_events()
                time.sleep(timer)
                marker_x_value.set_xdata(x_value)
                marker_x_value.set_ydata(0)
                figure.canvas.flush_events()
                time.sleep(timer)
                marker_y_value.set_xdata(x_value)
                marker_y_value.set_ydata(y_value)
                figure.canvas.flush_events()
                time.sleep(timer)
                marker_samples.set_xdata(samples[:, 0])
                marker_samples.set_ydata(samples[:, 1])
                figure.canvas.flush_events()
                time.sleep(timer)
                figure.canvas.draw() # drawing updated values

                # This will run the GUI event loop until all UI events currently waiting have been processed
                figure.canvas.flush_events()
                time.sleep(timer)

            if accept:
                x_value = sample[0]
                y_value = sample[1]

        return samples, self.acceptances, self.rejections


#Run Sampler
begin_time = datetime.datetime.now()

samples_n = 1000
burnin = 200
mu = [15, 20]
sigma = [1, 3]
x_range = [5, 35]
step_size = 0.9
seed = 0

sampler = MetropolisHastingsSampler(mu, sigma, x_range, step_size, seed)
samples, acceptances, rejections = sampler.sample_1d(samples_n, burnin, plot=False)
samples_accepted = samples[~np.all(samples == 0, axis=1)]

# Plot End Result
x = np.linspace(x_range[0], x_range[1], 1000)
target_distribution = np.zeros(len(x))
for idx, val in enumerate(x):
    target_distribution[idx] = sampler.p(x[idx], mu, sigma)

target_distribution_norm = [float(i)/sum(target_distribution) for i in target_distribution]

plt.figure(figsize=(17, 6))
plt.hist(samples_accepted[:, 0], bins=30, color='b', density=True, alpha=0.6)
plt.plot(x, target_distribution, '-r', linewidth=2)
plt.title("Metropolis Hastings Sampling", fontsize=16)
plt.xlim(x_range)
plt.ylim([-0.1, 0.75])
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

# Momentums
samples_mean = statistics.mean(samples_accepted[:, 0])
samples_std = statistics.stdev(samples_accepted[:, 0])
target_mean = sum(np.multiply(x, target_distribution_norm))
target_std = math.sqrt(sum(np.multiply(target_distribution_norm, (x-target_mean)**2)))

print('Acc: {} / Rej: {} / Ratio: {}'.format(acceptances, rejections, acceptances/rejections))
print('Mean of Samples: {}'.format(samples_mean))
print('Mean of Target: {}'.format(target_mean))
print('Standard Deviation of Samples: {}'.format(samples_std))
print('Standard Deviation of Target: {}'.format(target_std))
print('Standard Error: {}'.format((samples_std/samples_n)**0.5))
print('Execution Time: {}'.format(datetime.datetime.now() - begin_time))

__source__ = ''