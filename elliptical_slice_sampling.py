"""
Script for Elliptical Slice Sampling
"""

# Import
from os import fpathconf
import faulthandler
import numpy as np
from scipy.stats import norm
from numpy.core.fromnumeric import mean
import matplotlib.pyplot as plt
import imageio

class EllipticalSliceSampler:
    def __init__(self, mean, covariance, log_likelihood_func):
        self.mean = mean
        self.covariance = covariance
        self.function_likelihood_log = log_likelihood_func
        faulthandler.enable()

    def __sample(self, f, number):
        #nu = np.random.multivariant_normal(self.mean, self.covariance)
        nu = np.random.multivariate_normal(
            np.zeros(self.mean.shape), self.covariance)
        log_y = self.function_likelihood_log(f) + np.log(np.random.uniform())
        theta = np.random.uniform(0., 2.*np.pi)
        theta_min, theta_max = theta - 2.*np.pi, theta

        run = 0;

        while True:
            fp = (f - self.mean)*np.cos(theta) + nu*np.sin(theta) + self.mean

            a = np.linspace(0, 2*np.pi, num=1000)
            x = [(f-self.mean)*np.cos(i) for i in a]
            y = [nu * np.sin(i) + self.mean for i in a]

            slice = np.linspace(theta_min,theta_max, num=1000)
            x_slice = [(f-self.mean)*np.cos(i) for i in slice]
            y_slice = [nu * np.sin(i) + self.mean for i in slice]

            xy = zip(x,y)
            xy = [a for a in xy if self.function_likelihood_log(a[0]+a[1]) > log_y]

            fp_x = (f - self.mean)*np.cos(theta)
            fp_y = nu*np.sin(theta) + self.mean

            print(number)

            plt.scatter(fp_x, fp_y, color='r', zorder=2, marker='x', s=100)
            plt.scatter(x,y, zorder=0, color='k')
            # plt.scatter(x_slice, y_slice,color='g',zorder=1)
            for bracket_point in xy:
                plt.scatter(bracket_point[0], bracket_point[1],color='g',zorder=1)
            plt.title('sample '+str(number))
            plt.savefig('./gif/'+str(number)+'_'+str(run)+'.png')
            # plt.show()
            plt.close()

            log_fp = self.function_likelihood_log(fp)
            if log_fp > log_y:
                return fp
            else:
                run += 1
                if theta < 0.:
                    theta_min = theta
                else:
                    theta_max = theta
                theta = np.random.uniform(theta_min, theta_max)

    def sample(self, n_samples, burnin=1000):
        total_samples = n_samples + burnin
        samples = np.zeros((total_samples, self.covariance.shape[0]))
        samples[0] = np.random.multivariate_normal(
            mean=self.mean, cov=self.covariance)

        for i in range(1, total_samples):
            samples[i] = self.__sample(samples[i-1], i)
        return samples[burnin:]


def main():
    import numpy as np
    from scipy.stats import norm
    np.random.seed(0)

    mu_1 = 5.
    mu_2 = 1.
    sigma_1, sigma_2 = 1., 2.
    mu = ((sigma_1**-2)*mu_1 + (sigma_2**-2)*mu_2) / \
        (sigma_1**-2 + sigma_2**-2)
    sigma = np.sqrt((sigma_1**2 * sigma_2**2) / (sigma_1**2 + sigma_2**2))

    def log_likelihood_func(f):
        return norm.logpdf(f, mu_2, sigma_2)

    n_samples = 100
    sampler = EllipticalSliceSampler(np.array([mu_1]), np.diag(
        np.array([sigma_1**2, ])), log_likelihood_func)

    samples = sampler.sample(n_samples=n_samples, burnin=0)

    r = np.linspace(0., 8., num=100)
    plt.figure(figsize=(17, 6))
    plt.hist(samples, bins=30, color='b', density=True, alpha=0.6)
    # plt.plot(r, norm.pdf(r, mu, sigma))
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin,xmax, 100)
    p = norm.pdf(x, mu, sigma)
    plt.plot(x, p, 'k', linewidth=2)
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()

__author__ = 'Viking Penguin'
__source__ = 'https://www.youtube.com/watch?v=HfzyuD9_gmk&t=448s'
