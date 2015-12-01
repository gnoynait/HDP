import numpy as np
import scipy.special as sp
from scipy import linalg
import os, sys, math, time
from scipy.spatial import distance
from itertools import izip
import random
import cPickle
from sklearn import cluster

def log_normalize(v):
    ''' return log(sum(exp(v)))'''
    log_max = 100.0
    max_val = np.max(v, 1)
    log_shift = log_max - np.log(v.shape[1]+1.0) - max_val
    tot = np.sum(np.exp(v + log_shift[:,np.newaxis]), 1)

    log_norm = np.log(tot) - log_shift
    v = v - log_norm[:,np.newaxis]

    return (v, log_norm)

class FullFactorSpheGaussianMixture:
    def __init__(self, k, dim, gamma0, a0, b0):
        """
        precision: lambda ~ Gamma(a, b)
        mean: mu ~ Norm(nu, 1/(gamma*lambda))
        """
        self.dim = dim
        self.k = k
        self.gamma0, self.a0, self.b0 = gamma0, a0, b0
        self.gamma = np.ones(k) * gamma0
        self.nu = np.random.randn(k, dim) * 5
        self.a = np.ones(k) * a0
        self.b = np.ones(k) * b0
        self._updateExpectation()

    def update(self, X, z, scale, lr):
        stat0 = scale * np.sum(z, axis=0)
        stat1 = scale * np.dot(z.T, X)
        stat2 = scale * np.sum(z * np.sum(np.square(X), axis=1)[:, np.newaxis], axis=0)
        gamma = lr * (self.gamma0 + stat0) + (1-lr) * self.gamma
        a2minus1 = lr * (self.a0 * 2 - 1 + stat0) + (1-lr) * (self.a * 2 - 1)
        gammanu = lr * stat1 + (1-lr) * self.gamma[:, np.newaxis] * self.nu
        gammasqnuplus2b = lr * (stat2 + self.b0 * 2) \
            + (1-lr) * (self.gamma * np.sum(np.square(self.nu), axis=1) + 2 * self.b)

        self.gamma = gamma
        self.a = 0.5 * (a2minus1 + 1)
        self.nu = gammanu / gamma[:, np.newaxis]
        self.b = 0.5*(gammasqnuplus2b - np.sum(np.square(gammanu), axis=1) / gamma)

        self._updateExpectation()

    def _updateExpectation(self):
        self.expc_mu = self.nu
        self.expc_lnlambda = (sp.psi(self.a) - np.log(self.b)) / self.dim
        self.expc_lambda = self.a / self.b
        self.expc_lambdasqmu = self.expc_lambda * np.sum(np.square(self.nu), axis=1)\
            + 1.0 / self.gamma

    def calcLogProb(self, X):
        sqX = np.sum(np.square(X), axis=1)
        muX = np.sum(self.expc_mu[np.newaxis,:,:] * X[:,np.newaxis,:], axis=2)
        logprob = -0.5 * self.expc_lambda * sqX[:,np.newaxis] \
            + self.expc_lambda * muX\
            + 0.5 * self.expc_lambdasqmu\
            + 0.5 * self.dim * self.expc_lnlambda[np.newaxis, :]\
            - 0.5 * self.dim * np.log(2 * np.pi)
        return logprob

    def entropy(self):
        d = self.dim
        ents = 0.5 * d * (1 + np.log(2*np.pi/self.gamga)) + sp.gammaln(a) + a\
            - 0.5 * (2 * a - 2 + d) * sp.psi(a) + 0.5 * (d - 2) * np.log(b)
        return np.sum(ents)

    def expectLogPrior(self):
        logp = -0.5 * self.dim * np.log(2 * np.pi) \
            - 0.5 * self.gamma0 * self.expc_lambda * (np.sum(np.square(self.nu), axis=1)) \
            - 0.5 * self.gamma0 / self.gamma + 0.5 * np.log(self.gamma0) \
            - self.bo * self.expc_lambda + (self.a0 - 0.5) * self.expc_lnlambda\
            + self.a0 * np.log(self.b0)-sp.gammaln(self.a0)
        return np.sum(logp)

class NonBayesianWeight:
    def __init__(self, T):
        self.T = T
        self.weight = np.ones(T) / T

    def logWeight(self):
        return np.log(self.weight)

    def update(self, z, scale, lr):
        z = z / np.sum(z)
        self.weight = lr * z + (1-lr) * self.weight
    def entropy(self):
        return 0.0

class DirichletWeight:
    def __init__(self, T, alpha):
        self.T = T
        self.alpha = alpha
        self.weight = np.ones(T) * alpha

    def logWeight(self):
        return np.log(self.weight / np.sum(self.weight))

    def update(self, z, lr):
        self.weight = lr * (np.log(z) + self.alpha) + (1.0-lr) * self.weight
    def entropy(self):
        s = np.sum(self.weight)
        b = np.sum(sp.gammaln(self.weight)) - sp.gammaln(s)
        ent = b + (s - self.T) * sp.psi(s) \
            - np.sum((self.weight-1)*sp.psi(self.weight))
        return ent

class StickBreakingWeight:
    def __init__(self, T, alpha):
        """
        T: trunction level
        alpha: concentration parameter
        """
        self.T = T
        self.alpha = alpha
        sticks = np.zeros((2, T-1))
        sticks[0] = 1.0
        sticks[1] = alpha

        self.sticks = sticks
        self._calcLogWeight()

    def _calcLogWeight(self):
        """E[log(sticks)] 
        """
        sticks = self.sticks
        dig_sum = sp.psi(np.sum(sticks, 0))
        ElogW = sp.psi(sticks[0]) - dig_sum
        Elog1_W = sp.psi(sticks[1]) - dig_sum

        n = len(sticks[0]) + 1
        Elogsticks = np.zeros(n)
        Elogsticks[0:n-1] = ElogW
        Elogsticks[1:] = Elogsticks[1:] + np.cumsum(Elog1_W)
        self.expc_logw =  Elogsticks 
        
    def logWeight(self):
        """E[log(sticks)"""
        return self.expc_logw

    def update(self, z, scale, lr):
        """
        z: (n, T) numpy ndarray
        lr: float
        """
        z = np.sum(z, axis=0)
        stick0 = scale * z[:self.T-1] + 1.0
        stick1 = scale * np.flipud(np.cumsum(np.flipud(z[1:]))) + self.alpha
        self.sticks[0] = lr * stick0 + (1.0 - lr) * self.sticks[0]
        self.sticks[1] = lr * stick1 + (1.0 - lr) * self.sticks[1]
        self._calcLogWeight()

    def entropy(self):
        a, b = self.sticks[0], self.sticks[1]
        ents = a - np.log(b) + sp.gammaln(a) + (1 - a)*sp.psi(a)
        return np.sum(ents)

    def expectLogPrior(self):
        logp = np.log(self.alpha) - self.alpha * self.sticks[0] / self.sticks[1]
        return np.sum(p)

class DPMixture:
    """Online DP model"""
    def __init__(self, T, dim, model, weight):
        self.model = model
        self.weight = weight

    def assign(self, X):
        Eloggauss = self.model.calcLogProb(X)
        z = Eloggauss + self.weight.logWeight()
        z, _= log_normalize(z)
        z = np.exp(z)
        return z

    def predict(self, X):
        logLik = self.model.calcLogProb(X)
        post = logLik + self.weight.logWeight()
        return post.argmax(axis=1)

    def update(self, X, scale, lr):
        z = self.assign(X)
        self.weight.update(z, scale, lr)
        self.model.update(X, z, scale, lr)

    def logLikelihood(self, X, scale):
        Eloggauss = self.model.calcLogProb(X)
        z = Eloggauss + self.weight.logWeight()
        z, _= log_normalize(z)
        z = np.exp(z)
        likelihood = np.sum(z * Eloggauss, axiz=(0,1)) + np.sum(z * self.weight.logWeight()[np.newaxis,:])
        likelihood += self.weight.entropy() + self.model.entropy()\
                + self.weight.expectLogPrior() + self.model.expectLogPrior()
        return likelihood


class DecaySheduler:
    def __init__(self, tau, kappa, minlr):
        ## for online learning
        self.tau = tau
        self.kappa = kappa
        self.count = 0 # update count
        self.minlr = minlr

    def nextRate(self):
        lr = pow(self.tau + self.count, -self.kappa)
        if lr < self.minlr:
            lr = self.minlr
        self.count += 1
        return lr

class ConstSheduler:
    def __init__(self, learningRate):
        self.lr = learningRate
    def nextRate(self):
        return self.lr
        
class Trainer:
    def __init__(self, model, lrSheduler):
        self.sheduler = lrSheduler
        self.model = model
    def fit(self, X, n):
        for i in range(n):
            self.model.update(X, 1.0, self.sheduler.nextRate())

