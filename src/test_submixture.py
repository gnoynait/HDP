import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm

from sklearn import datasets
import model as md

n_samples = 50000
std = 0.5
centers = [(-2.5, -2.5), (2.5, 2.5), (-2.5, 2.5), (2.5, -2.5)]
#centers = [(-2.5+10, -2.5), (2.5+10, 2.5), (-2.5+10, 2.5), (2.5+10, -2.5)]
X, _ = datasets.make_blobs(n_samples=n_samples, n_features=2, cluster_std=std,
                  centers=centers, shuffle=True, random_state=None)
sheduler = md.DecaySheduler(100, 0.6, 0.001)
sheduler = md.ConstSheduler(1.0)
#base = md.FullFactorSpheGaussianMixture(100, 2, 1, 1000000, 1, X)
base = md.StandardGaussianMixture(100, 2, 1, 1e10, X)
model = md.SubDPMixture(100, 100, base)
for i in range(500):
    step = n_samples
    count = n_samples / step
    for iter in range(count):
        model.update(X[iter * step: (iter+1) * step, :], 10000000 * count, sheduler.nextRate())
y = model.predict(X)
y_unique = np.unique(y)
colors = cm.rainbow(np.linspace(0.0, 1.0, y_unique.size))
for this_y, color in zip(y_unique, colors):
    this_X = X[y == this_y]
    plt.scatter(this_X[:, 0], this_X[:, 1], c=color, alpha=0.5,
                    label="Class %s" % (this_y))
    if isinstance(base, md.FullFactorSpheGaussianMixture):
        mu = base.expc_mu[this_y]
        lmbd = base.expc_lambda[this_y]
        stdvar = np.sqrt(1/ lmbd)
    else:
        mu = base.mu[this_y]
        stdvar = std
    axis = plt.gca()
    ell = mpl.patches.Ellipse(mu, stdvar, stdvar, color=color)
    ell.set_alpha(0.5)
    axis.add_artist(ell)

plt.legend(loc="best")
plt.title("Data")

plt.show()
