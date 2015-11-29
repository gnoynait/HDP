import numpy as np
from sklearn.datasets import make_blobs
class RandomBlob:
    def __init__(self, dim):
        self.centers = centers = [(-5, -5), (0, 0), (5, 5)]
        self.dim = dim
    def nextBatch(self, n):
        X, _ = make_blobs(n_samples=n, n_features=self.dim, cluster_std=1.0,
                          centers=self.centers, shuffle=True)
        return X

class FixSizeData:
    def __init__(self, n, dim):
        self.n = n
        self.dim = dim
        self.count = 0
    def beforeFirst(self):
        self.count = 0
        self.data = RandomBlob(self.dim).nextBatch(self.n)
    def nextBatch(self, bat):
        if self.count >= self.n:
            return 0
        self.count += 1
        return bat
    def value(self):
        return self.data

