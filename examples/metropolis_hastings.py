"""
Script for Metropolis Hastings Sampling
"""

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
        self.accepted = 0
        self.rejected = 0
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

    def q_norm(self, x_prior, sigma):
        """
        Defines a 1D Gaussian Proposal Distribution
        :param x: x value
        :param mu: List of Expected Values
        :param sigma: List of Standard Deviations
        :return:
        """
        x_proposal = np.random.normal(x_prior, sigma, 1)
        q_ab = st.norm.pdf(x_proposal, x_prior, scale=sigma)
        q_ba = st.norm.pdf(x_prior, x_proposal, scale=sigma)
        return x_proposal, q_ab, q_ba

    def q_gumbel_l(self, x_prior):
        """
        Defines a 1D Left Skewed Gumbel Proposal Distribution
        :param x: x value
        :param mu: List of Expected Values
        :param sigma: List of Standard Deviations
        :return:
        """
        x_proposal = np.random.gumbel(x_prior, 1, 1)
        q_ab = st.gumbel_l.pdf(x_proposal, x_prior, scale=1)
        q_ba = st.gumbel_l.pdf(x_prior, x_proposal, scale=1)
        return x_proposal, q_ab, q_ba

    def sample(self, x_prior, y_prior):
        """
        Creates a single Sample. x_value is the prior x
        :param x_value: float
        :param y_value: float
        :return: array sample
        """

        # Propose Sample
        sigma_total = np.sqrt((sigma[0] ** 2 * sigma[1] ** 2) / (sigma[0] ** 2 + sigma[1] ** 2))
        x_proposed, q_ab, q_ba = sampler.q_norm(x_prior, sigma_total) # This is symmetric, so metropolis
        #x_proposed, q_ab, q_ba = sampler.q_gumbel_l(x_prior) # This is asymmetric, so metropolis-hastings

        #TODO: Paper
        #mu_total = ((sigma[0] ** -2) * mu[0] + (sigma[1] ** -2) * mu[1]) / (sigma[0] ** -2 + sigma[1] ** -2)
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

        rf = p_proposed/p_prior
        rg = q_ab/q_ba

        acceptance_ratio = rf * rg
        acceptance_probability = min(1, acceptance_ratio)
        logging.debug('Acceptance Probability: {}'.format(acceptance_probability))

        sample = [0, 0]
        if np.random.rand() < acceptance_probability:
            accept = True
            self.accepted += 1
            logging.debug('Sample Accepted: [{} {}]'.format(x_proposed, y_proposed))
            sample[0] = x_proposed
            sample[1] = y_proposed
        else:
            accept = False
            self.rejected += 1
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
        samples = np.zeros([samples_n, 2])

        # Set up Plotting
        if plot:
            plt.ion()
            figure, ax = plt.subplots(figsize=(10, 8))

        # Create Samples
        x_value = np.random.uniform(self.x_range[0], self.x_range[1])  # Random x for first sample
        y_value = 0.                                                   # y for first sample
        for i in range(samples_n):
            sample, accept = sampler.sample(x_value, y_value)
            samples[i, 0] = sample[0]
            samples[i, 1] = sample[1]

            # Update Plot
            if plot:
                timer = 0.0
                ax.cla()
                ax.plot(x_vector, target_distribution, '-', linewidth=2, markersize=1)
                ax.plot(samples[:, 0], samples[:, 1], 'xk', linewidth=2, markersize=10)
                plt.hist(samples[:, 0], bins=30, color='g', density=True, alpha=0.6)
                plt.title("Metropolis Hastings Sampling", fontsize=16)
                plt.text(5.5, 0.4, 'Iteration: {} \nAccepted: {} \nRejected: {}'.format(i, self.accepted, self.rejected), fontsize=16)
                plt.xlim(x_range)
                plt.ylim([-0.1, 0.5])
                plt.xlabel("X")
                plt.ylabel("Y")
                plt.legend(['Target Distribution', 'Samples', 'Histogram'])
                time.sleep(timer)
                figure.canvas.flush_events()
                figure.canvas.draw()

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
samples, acceptances, rejections = sampler.sample_1d(samples_n, burnin, plot=True)
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