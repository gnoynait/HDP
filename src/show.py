import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from sklearn.datasets import make_blobs
from sklearn.metrics import brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cross_validation import train_test_split
import model as md


n_samples = 5000
n_bins = 3  # use 3 bins for calibration_curve as we have 3 clusters here

centers = [(-5, 15), (0, 0), (5, 8)]
X, y = make_blobs(n_samples=n_samples, n_features=2, cluster_std=.5,
                  centers=centers, shuffle=False, random_state=42)
T = 10
dim = 2
alpha = 1 
model = md.DPMixture(T, dim, md.FullFactorSpheGaussianMixture(T, dim, 1, 4, 2),
    md.StickBreakingWeight(T, alpha))
    #md.NonBayesianWeight(T))
sheduler = md.DecaySheduler(10, 0.5, 0.1)
updater = md.Trainer(model, sheduler)
updater.fit(X, 100)
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
