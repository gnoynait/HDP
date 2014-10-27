import itertools

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn import mixture
from onlinedpgmm import *
from sklearn.externals.six.moves import xrange
# Number of samples per component
n_samples = 5000


T = 50
K = 20 
topics = 10
D = 500
alpha = 1 # second level
gamma = 2 # first level
kappa = 0.75
tau = 1
dim = 2
total = 100000

# Generate random sample following a sine curve
np.random.seed(1)
X = np.zeros((n_samples, 2))
step = 4 * np.pi / n_samples

def gen_grid_data(n):
	width, height = 5, 5
	comp1 = int(np.random.uniform() * n)
	comp2 = n - comp1
	means = np.array(np.random.uniform(size = (2, 2)) * 5, dtype = "int") - 2
	x1 = np.random.multivariate_normal(means[0], np.eye(2) * 0.05, comp1)
	x2 = np.random.multivariate_normal(means[1], np.eye(2) * 0.05, comp2)
	x = np.vstack((x1, x2))
	np.random.shuffle(x)
	return x
	

for i in xrange(X.shape[0]):
    x = i * step - 6
    X[i, 0] = x + np.random.normal(0, 0.1)
    X[i, 1] = 3 * (np.sin(x) + np.random.normal(0, .2))


color_iter = itertools.cycle(['r', 'g', 'b', 'c', 'm'])


for i, (clf, title) in enumerate([
#        (mixture.GMM(n_components=10, covariance_type='full', n_iter=100),
#         "Expectation-maximization"),
#        (mixture.DPGMM(n_components=10, covariance_type='full', alpha=0.01,
#                       n_iter=100),
#         "Dirichlet Process,alpha=0.01"),
#        (mixture.DPGMM(n_components=10, covariance_type='diag', alpha=100.,
#                       n_iter=100),
#         "Dirichlet Process,alpha=100.")]):
        (online_hdp(T, K, D, alpha, gamma, kappa, tau, total, dim, 'diagonal'), "online hdp"), 
        (online_dp(T, gamma, kappa, tau, total, dim, 'diagonal'), "online dp")]):

    splot = plt.subplot(2, 1, 1 + i)
    if True:
        clf.fit(X, 200, 200)
    else:
        X = np.array([0, 0], dtype = 'float64')
        for c in range(200):
            print c
            d = gen_grid_data(100)
            X = np.vstack((X, d[:10,:]))
            clf.process_documents([d, gen_grid_data(100), gen_grid_data(100), gen_grid_data(100)])
    Y_ = clf.predict(X)
    for i, (mean, cov, color) in enumerate(zip(
            clf.m_means, clf.get_cov(), color_iter)):
        v, w = linalg.eigh(cov)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        #plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / (u[0] + np.finfo(np.float32).eps))
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
