import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

mu_1 = 0.
mu_2 = 4.
sigma_1, sigma_2 = 1., 2.
mu = ((sigma_1**-2)*mu_1 + (sigma_2**-2)*mu_2) / \
    (sigma_1**-2 + sigma_2**-2)
sigma = np.sqrt((sigma_1**2 * sigma_2**2) / (sigma_1**2 + sigma_2**2))


f_size = (10, 6)


r = np.linspace(-4., 8., num=1000)

c_target = 'b'
c_likeli = 'r'
c_norm = 'orange'

nu = 1
f = 4
L_f = norm.pdf(f, mu_2, sigma_2)
f_p = 0
L_fp = norm.pdf(f_p, mu_2, sigma_2)
y = L_f * 0.6


def plot_gen_pic(f_size, c_target, c_likeli, c_norm):
    plt.figure(figsize=(f_size))
    plt.plot(r, norm.pdf(r, mu, sigma),
             label='target distribution', c=c_target)
    plt.plot(r, norm.pdf(r, mu_1, sigma_1),
             label='$\mathcal{N}(0, \Sigma)$', c=c_norm)
    plt.plot(r, norm.pdf(r, mu_2, sigma_2), label='likelihood', c=c_likeli)
    plt.plot(nu, 0, 'v', markersize=12, c=c_norm)
    plt.grid()
    plt.legend()


# step 1 - select nu
plot_gen_pic(f_size, c_target, c_likeli, c_norm)

plt.title(r'step 1 - select $\nu$')
plt.plot(f, 0, 'o', c=c_target)                 # plot f as initial state
plt.scatter(nu, 0, c=c_norm)
plt.savefig('ess_step_01.png', dpi=300)
plt.show()

# step 2 - calculate y
plot_gen_pic(f_size, c_target, c_likeli, c_norm)
plt.title(r'step 2 - compute log-likelihood')
plt.plot(f, 0, 'o', c=c_target)
plt.vlines([f], 0, L_f, linestyles='dashed', colors=c_likeli)
plt.scatter(f, L_f, c=c_likeli)
plt.scatter(f, y,  c=c_likeli)
plt.scatter(nu, 0, c=c_norm)
plt.annotate("$L(f)$", xy=(f, L_f), xytext=(
    f+0.5, L_f+0.1), arrowprops=dict(arrowstyle="->"))
plt.annotate("$\log y = \log L(f) + \log u $", xy=(f, y),
             xytext=(f+0.5, y+0.1), arrowprops=dict(arrowstyle="->"))

plt.hlines(y=y, xmin=-4, xmax=8)
plt.show()

# step 3
plot_gen_pic(f_size, c_target, c_likeli, c_norm)
plt.title(r'step 3 - compute $f´$')
plt.scatter(f, 0, c=c_target)

##
plt.scatter(f, L_f, c=c_likeli)
##
plt.scatter(f, y,  c=c_likeli)
plt.hlines(y=y, xmin=-4, xmax=8, linestyles='dashed')
plt.hlines(y=y, xmin=2, xmax=6, colors='g', linewidth=4)
##
plt.plot(f_p, L_fp, '*', markersize=12, c=c_likeli)
plt.plot(f_p, 0, '*', markersize=12, c=c_likeli)
plt.plot(f_p, y, '*', markersize=12, c=c_likeli)
plt.vlines([f_p], 0, L_fp, linestyles='dashed', colors=c_likeli)

#plt.vlines([f], 0, L_f, linestyles='dashed', colors=c_likeli)
#
# plt.annotate("$L(f)$", xy=(f, L_f), xytext=(
#    f+0.5, L_f+0.1), arrowprops=dict(arrowstyle="->"))
# plt.annotate("$\log y = \log L(f) \cdot \log u $", xy=(f, y),
#             xytext=(f+0.5, y+0.1), arrowprops=dict(arrowstyle="->"))

plt.show()


# step 4
f_p = 5
L_fp = norm.pdf(f_p, mu_2, sigma_2)
plot_gen_pic(f_size, c_target, c_likeli, c_norm)
plt.title(r'step 3 - compute $f´$')
plt.scatter(f, 0, c=c_target)

##
plt.scatter(f, L_f, c=c_likeli)
##
plt.scatter(f, y,  c=c_likeli)
plt.hlines(y=y, xmin=-4, xmax=8, linestyles='dashed')
plt.hlines(y=y, xmin=2, xmax=6, colors='g', linewidth=4)
##
plt.plot(f_p, L_fp, '*', markersize=12, c=c_likeli)
plt.plot(f_p, 0, '*', markersize=12, c=c_likeli)
plt.plot(f_p, y, '*', markersize=12, c=c_likeli)
plt.vlines([f_p], 0, L_fp, linestyles='dashed', colors=c_likeli)

plt.show()
