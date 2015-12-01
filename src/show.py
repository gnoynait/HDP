import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from sklearn.datasets import make_blobs
from sklearn.metrics import brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cross_validation import train_test_split
import model as md

np.random.seed()

n_samples = 5000

centers = [(-2.5, 2.5), (2.5, -2.5), (5.0, 0)]
X, y = make_blobs(n_samples=n_samples, n_features=2, cluster_std=.5,
                  centers=centers, shuffle=True, random_state=42)
T = 10 
dim = 2
alpha = 1
gamma = 1e-200
a = 1000
b = 1000

base = md.FullFactorSpheGaussianMixture(T, dim, gamma, a, b)
#base = md.StandardGaussianMixture(T, dim, gamma)
weight = md.StickBreakingWeight(T, alpha)
#weight = md.NonBayesianWeight(T)
model = md.DPMixture(T, dim, base, weight)
    
sheduler = md.ConstSheduler(1)
sheduler = md.DecaySheduler(1, 0.5, 0.001)
updater = md.Trainer(model, sheduler)
updater.fit(X, 100)
print np.exp(weight.logWeight()), base.expc_lambda, base.expc_mu
for w, lmbd, mu in zip(np.exp(weight.logWeight()), base.expc_lambda, base.expc_mu):
    print w, lmbd, mu
y = model.predict(X)
y_unique = np.unique(y)
colors = cm.rainbow(np.linspace(0.0, 1.0, y_unique.size))
for this_y, color in zip(y_unique, colors):
    this_X = X[y == this_y]
    plt.scatter(this_X[:, 0], this_X[:, 1], c=color, alpha=0.5,
                    label="Class %s" % this_y)
plt.legend(loc="best")
plt.title("Data")

plt.show()
