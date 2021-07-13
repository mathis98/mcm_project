"""
Slice Sampling
"""

# TODO: Add 2d Case

# Imports
import numpy as np
import scipy.stats as st
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import time
import statistics
import math


class SliceSampler:
    def __init__(self, mu=[1], sigma=[1], x_range=[-5, 5], step_size=2.5, seed=0):
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

    def sample_1d(self, samples_n, plot = False):
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
            # Get Y in [0 p(x)]
            y_range = [0, sampler.p(x_value, self.mu, self.sigma)]
            y_vector = np.linspace(y_range[0], y_range[1], 2)

            # Get random y
            y_value = np.random.uniform(y_range[0], y_range[1])

            # Create Horizontal Line w with Step Size
            w_length = self.step_size
            w_lower_end = np.random.uniform(x_value -  w_length, x_value)
            w_range = [w_lower_end, w_lower_end +  w_length]
            w_vector = np.linspace(w_range[0], w_range[1], 2)

            # Extent w until a fitting step size is found / Adjust Step Size
            while y_value < sampler.p(w_range[1], mu, sigma):  # To the right
                w_range[1] +=  w_length

            while y_value < sampler.p(w_range[0], mu, sigma):  # To the left
                w_range[0] -=  w_length

            w_vector = np.linspace(w_range[0], w_range[1], 2)

            # Sample value
            sample_found = False
            while not sample_found:
                x_sample = np.random.uniform(w_range[0], w_range[1])
                if y_value < sampler.p(x_sample, mu, sigma):
                    sample_found = True
                    samples[i, 0] = x_sample
                    samples[i, 1] = y_value

            if plot:
                # updating data values
                timer = 0.001
                line_distribution.set_xdata(x_vector)
                line_distribution.set_ydata(target_distribution)
                figure.canvas.flush_events()
                time.sleep(timer)
                marker_x_value.set_xdata(x_value)
                marker_x_value.set_ydata(0)
                figure.canvas.flush_events()
                time.sleep(timer)
                line_y_vector.set_xdata([x_value] * len(y_vector))
                line_y_vector.set_ydata(y_vector)
                figure.canvas.flush_events()
                time.sleep(timer)
                marker_y_value.set_xdata(x_value)
                marker_y_value.set_ydata(y_value)
                figure.canvas.flush_events()
                time.sleep(timer)
                line_w.set_xdata(w_vector)
                line_w.set_ydata([y_value] * len(w_vector))
                figure.canvas.flush_events()
                time.sleep(timer)
                marker_samples.set_xdata(samples[:, 0])
                marker_samples.set_ydata(samples[:, 1])
                figure.canvas.flush_events()
                time.sleep(timer)

                # drawing updated values
                figure.canvas.draw()

                # This will run the GUI event
                # loop until all UI events
                # currently waiting have been processed
                figure.canvas.flush_events()
                time.sleep(timer)

            x_value = x_sample

        return samples

samples_n = 10000
mu = [15, 20]
sigma = [1, 3]
x_range = [0, 50]
w_length = 0.5
seed = 0

sampler = SliceSampler(mu, sigma, x_range, w_length, seed)
samples = sampler.sample_1d(samples_n, plot=True)

# Plot End Result
x = np.linspace(x_range[0], x_range[1], 1000)
target_distribution = np.zeros(len(x))
for idx, val in enumerate(x):
    target_distribution[idx] = sampler.p(x[idx], mu, sigma)

target_distribution_norm = [float(i)/sum(target_distribution) for i in target_distribution]

plt.figure(figsize=(17, 6))
plt.hist(samples[:, 0], bins=30, color='b', density=True, alpha=0.6)
plt.plot(x, target_distribution, '-r', linewidth=2)
plt.grid()
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

__source__ = 'https://wiseodd.github.io/techblog/2015/10/24/slice-sampling/'