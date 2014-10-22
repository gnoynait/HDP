import itertools

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn import mixture
from onlinedpgmm import *
from sklearn.externals.six.moves import xrange

# Number of samples per component
n_samples = 1000


T = 50 
K = 25 
topics = 10
D = 500
alpha = 0.1 # second level
gamma = 1 # first level
kappa = 0.95
tau = 10
dim = 2
total = 10000

# Generate random sample following a sine curve
#random_seed = int(time.time())
random_seed = 1
np.random.seed(random_seed)
X = np.zeros((n_samples, 2))
step = 4 * np.pi / n_samples

for i in xrange(X.shape[0]):
    x = i * step - 6
    X[i, 0] = x + np.random.normal(0, 0.1)
    X[i, 1] = 3 * (np.sin(x) + np.random.normal(0, .2))


color_iter = itertools.cycle(['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w'])


for i, (clf, title) in enumerate([
#        (mixture.GMM(n_components=10, covariance_type='full', n_iter=100),
#         "Expectation-maximization"),
#        (mixture.DPGMM(n_components=10, covariance_type='full', alpha=0.01,
#                       n_iter=100),
#         "Dirichlet Process,alpha=0.01"),
#        (mixture.DPGMM(n_components=10, covariance_type='diag', alpha=100.,
#                       n_iter=100),
#         "Dirichlet Process,alpha=100.")]):
        (online_dp(T, K, D, alpha, gamma, kappa, tau, total, dim), "online dp")]):

    clf.fit(X)
    splot = plt.subplot(1, 1, 1 + i)
    Y_ = clf.predict(X)
    print clf.m_means
    for i, (mean, precis, color) in enumerate(zip(
            clf.m_means, clf.m_precis, color_iter)):
        v, w = linalg.eigh(linalg.inv(precis))
        #v, w = linalg.eigh(precis)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180 + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.xlim(-6, 4 * np.pi - 6)
    plt.ylim(-5, 5)
    plt.title(title)
    plt.xticks(())
    plt.yticks(())

plt.show()
