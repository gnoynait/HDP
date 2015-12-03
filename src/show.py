import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm

from sklearn import datasets
from sklearn.metrics import brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cross_validation import train_test_split
import model as md

n_samples = 50000
centers = [(-25, -25), (-25, 25), (25, -25), (25, 25)]
X, y = datasets.make_blobs(n_samples=n_samples, n_features=2, cluster_std=10,
                  centers=centers, shuffle=True, random_state=None)
#X, y = datasets.make_blobs(n_samples=n_samples, n_features=2, cluster_std=1.0, 
#                  centers=10, shuffle=True, random_state=None)
T = 20
dim = 2
alpha = 1
gamma = 0.001
a = 1
b = 100

base = md.FullFactorSpheGaussianMixture(T, dim, gamma, a, b)
#base = md.StandardGaussianMixture(T, dim, gamma, 0.001)
weight = md.StickBreakingWeight(T, alpha)
#weight = md.NonBayesianWeight(T)
model = md.DPMixture(T, dim, base, weight)
    
sheduler = md.DecaySheduler(1000, 0.7, 0.001)
updater = md.Trainer(model, sheduler)
loglik = updater.fit(X, 20, 100)
plt.plot(loglik)
plt.figure()
#print np.exp(weight.logWeight()), base.expc_lambda, base.expc_mu
#for w, lmbd, mu in zip(np.exp(weight.logWeight()), base.expc_lambda, base.expc_mu):
#    print w, lmbd, mu
y = model.predict(X)
y_unique = np.unique(y)
colors = cm.rainbow(np.linspace(0.0, 1.0, y_unique.size))
for this_y, color in zip(y_unique, colors):
    this_X = X[y == this_y]
    w = np.exp(weight.logWeight())[this_y]
    plt.scatter(this_X[:, 0], this_X[:, 1], c=color, alpha=0.5,
                    label="Class %s:%f" % (this_y, w))
    if isinstance(base, md.FullFactorSpheGaussianMixture):
        mu = base.expc_mu[this_y]
        lmbd = base.expc_lambda[this_y]
        stdvar = np.sqrt(1/ lmbd)
    else:
        mu = base.mu[this_y]
        stdvar = 1
    print mu, stdvar, w
    axis = plt.gca()
    ell = mpl.patches.Ellipse(mu, stdvar, stdvar, color=color)
    ell.set_alpha(0.5)
    axis.add_artist(ell)

plt.legend(loc="best")
plt.title("Data")

plt.show()
