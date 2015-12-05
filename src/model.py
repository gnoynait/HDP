import numpy as np
import scipy.special as sp

epsilon = 1e-50
def log_normalize(v):
    ''' return log(sum(exp(v)))'''
    log_max = 100.0
    max_val = np.max(v, 1)
    log_shift = log_max - np.log(v.shape[1]+1.0) - max_val
    tot = np.sum(np.exp(v + log_shift[:, np.newaxis]), 1)

    log_norm = np.log(tot) - log_shift
    v = v - log_norm[:, np.newaxis]

    return (v, log_norm)

class StandardGaussianMixture(object):
    def __init__(self, k, dim, gamma0, lmbd):
        self.k = k
        self.dim = dim
        self.lmbd = lmbd
        self.gamma0 = gamma0
        self.gamma = np.ones(k) * gamma0
        self.mu = np.zeros((k, dim))
    def update(self, X, z, scale, lr):
        stat0 = scale * np.sum(z, axis=0)
        stat1 = scale * np.dot(z.T, X)

        self.gamma = lr * (self.gamma0 + stat0) + (1-lr) * self.gamma

        self.mu = lr * stat1 / self.gamma[:, np.newaxis] + (1.0 - lr) * self.mu

    def logAssign(self, X):
        sqX = np.sum(np.square(X), axis=1) 
        muX = np.sum(self.mu[np.newaxis, :, :] * X[:, np.newaxis, :], axis=2)
        logprob = -0.5 * self.lmbd * sqX[:, np.newaxis] -0.5 / self.gamma\
            + self.lmbd * muX\
            + 0.5 * self.lmbd * np.sum(np.square(self.mu), axis=1)[np.newaxis, :]\
            - 0.5 * self.dim * np.log(2 * np.pi / self.lmbd)
        return logprob

class FullFactorSpheGaussianMixture(object):
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
        anat = lr * ((self.dim + 2 * self.a0 - 2) / self.dim + stat0)\
            + (1-lr) * ((self.dim-2+2*self.a)/self.dim)
        gammanu = lr * stat1 + (1-lr) * self.gamma[:, np.newaxis] * self.nu
        gammasqnuplus2b = lr * (stat2 + self.b0 * 2) \
            + (1-lr) * (self.gamma * np.sum(np.square(self.nu), axis=1) + 2 * self.b)

        self.gamma = gamma
        self.a = 0.5 * (anat * self.dim - self.dim + 2) 
        self.nu = gammanu / gamma[:, np.newaxis]
        self.b = 0.5*(gammasqnuplus2b - np.sum(np.square(gammanu), axis=1) / gamma)

        self._updateExpectation()

    def _updateExpectation(self):
        self.expc_mu = self.nu
        self.expc_lnlambda = sp.psi(self.a) - np.log(self.b) 
        self.expc_lambda = self.a / self.b
        self.expc_lambdasqmu = self.expc_lambda * np.sum(np.square(self.nu), axis=1)\
               + self.dim / self.gamma

    def logAssign(self, X):
        sqX = np.sum(np.square(X), axis=1)
        muX = np.sum(self.expc_mu[np.newaxis, :, :] * X[:, np.newaxis, :], axis=2)
        logprob = -0.5 * self.expc_lambda * sqX[:, np.newaxis] \
            + self.expc_lambda * muX\
            + 0.5 * self.expc_lambdasqmu\
            + 0.5 * self.dim * self.expc_lnlambda[np.newaxis, :]\
            - 0.5 * self.dim * np.log(2 * np.pi)
        return logprob

    def entropy(self):
        d = self.dim
        a, b = self.a, self.b
        ents = 0.5 * d * (1 + np.log(2*np.pi/self.gamma)) + sp.gammaln(a) + a\
            - 0.5 * (2 * a - 2 + d) * sp.psi(a) + 0.5 * (d - 2) * np.log(b)
        return np.sum(ents)

    def expectLogPrior(self):
        logp = -0.5 * self.dim * np.log(2 * np.pi) \
            - 0.5 * self.gamma0 * self.expc_lambda * (np.sum(np.square(self.nu), axis=1)) \
            - 0.5 * self.gamma0 / self.gamma + 0.5 * np.log(self.gamma0) \
            - self.b0 * self.expc_lambda + (self.a0 - 0.5) * self.expc_lnlambda\
            + self.a0 * np.log(self.b0)-sp.gammaln(self.a0)
        return np.sum(logp)

class StickBreakingWeight(object):
    def __init__(self, T, alpha):
        """
        T: trunction level
        alpha: concentration parameter
        """
        self.T = T
        self.alpha = alpha
        self.sticks = np.zeros((2, T-1))
        self.update(np.ones((1, T))/T, 100000, 1)
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
        self.expc_logw = Elogsticks 

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
        return np.sum(logp)

class DPMixture(object):
    """Online DP model"""
    def __init__(self, T, dim, model, weight):
        self.model = model
        self.weight = weight

    def logAssign(self, X):
        logLik = self.model.logAssign(X)
        return logLik + self.weight.logWeight() 

    def assign(self, X):
        logz = self.logAssign(X)
        logz, _ = log_normalize(logz)
        z = np.exp(logz)
        return z

    def predict(self, X):
        logz = self.logAssign(X)
        return logz.argmax(axis=1)

    def update(self, X, scale, lr):
        z = self.assign(X)
        self.weight.update(z, scale, lr)
        self.model.update(X, z, scale, lr)

    def logLikelihood(self, X, scale):
        Eloggauss = self.model.logAssign(X)
        z = Eloggauss + self.weight.logWeight()
        z, _ = log_normalize(z)
        z = np.exp(z)
        likelihood = np.sum(z * Eloggauss, axis=(0,1)) \
            + np.sum(z * self.weight.logWeight()[np.newaxis, :])
        likelihood += self.weight.entropy() + self.model.entropy()\
                + self.weight.expectLogPrior() + self.model.expectLogPrior()
        return likelihood

class SubDPMixture(object):
    """Online HDP model"""
    def __init__(self, K, dim, alpha, model):
        self.model = model
        self.phi = np.ones((K, K)) / K
        self.weight = StickBreakingWeight(K, alpha)

    def logAssign(self, X):
        logz = self.logAssign(X)
        logk = np.dot(logz, self.phi.T)
        logk = self.weight.logWeight() + logk
        logz = np.dot(logk, self.phi)
        return logz

    def assign(self, X):
        logz = self.logAssign(X)
        logz, _ = log_normalize(logz)
        z = np.exp(logz)
        return z

    def predict(self, X):
        logz = self.logAssign(X)
        return logz.argmax(axis=1)

    def update(self, X, scale, lr):
        logt = self.model.logAssign(X)
        t, _ = log_normalize(logt)
        t = np.exp(t)
        logk = np.dot(logt, self.phi.T) + self.weight.logWeight()
        k, _ = log_normalize(logk)
        k = np.exp(k)
        self.weight.update(k, scale, lr)
        newphi = np.dot(k.T, t)
        newt = np.dot(t.T, k)
        self.model.update(X, newt, scale, lr)
        self.phi = lr * newphi + (1 - lr) * newphi

    def logLikelihood(self, X, scale):
        loglik = 0
        loglik += self.weight.entropy() + self.weight.expectLogPrior()\
            + np.sum(np.dot(self.phi, self.model.weight.logWeight())) \
            + np.sum(self.phi * np.log(self.phi))
        return loglik


class DecaySheduler(object):
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

class Trainer(object):
    def __init__(self, model, lrSheduler):
        self.sheduler = lrSheduler
        self.model = model
    def fit(self, X, n, split):
        loglik = []
        for i in range(n):
            step = (X.shape[0] + split - 1)/ split
            for s in range(split):
                start = s * step
                end = min(start + step, X.shape[0])
                x = X[start:end, ...]
                self.model.update(x, split, self.sheduler.nextRate())
            loglik.append(self.model.logLikelihood(X, 1))
        return loglik

class NonBayesianWeight(object):
    def __init__(self, T):
        self.T = T
        self.weight = np.ones(T) / T

    def logWeight(self):
        return np.log(self.weight + epsilon)

    def update(self, z, scale, lr):
        z = np.sum(z, axis=0)
        z = z / np.sum(z)
        self.weight = lr * z + (1-lr) * self.weight
    def entropy(self):
        return 0.0

class DirichletWeight(object):
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

class ConstSheduler(object):
    def __init__(self, learningRate):
        self.lr = learningRate
    def nextRate(self):
        return self.lr
