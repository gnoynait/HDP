import numpy as np
import scipy.special as sp
from scipy import linalg
import os, sys, math, time
import utils
from scipy.spatial import distance
from itertools import izip
import random
import cPickle
from sklearn import cluster
import time
import sys

#random_seed = 999931111
random_seed = int(time.time())
np.random.seed(random_seed)
random.seed(random_seed)

## smallest rhot
rhot_bound = 0.0

def debug(*W):
    for w in W:
        sys.stderr.write(str(w) + '\n')
    sys.stderr.write('-' * 75 + '\n')

class NoSuchModeError(Exception):
    def __str__(self):
        return 'No Such Mode Error'

def log_normalize(v):
    ''' return log(sum(exp(v)))'''
    log_max = 100.0
    if len(v.shape) == 1:
        max_val = np.max(v)
        log_shift = log_max - np.log(len(v)+1.0) - max_val
        tot = np.sum(np.exp(v + log_shift))
        log_norm = np.log(tot) - log_shift
        v = v - log_norm
    else:
        max_val = np.max(v, 1)
        log_shift = log_max - np.log(v.shape[1]+1.0) - max_val
        tot = np.sum(np.exp(v + log_shift[:,np.newaxis]), 1)

        log_norm = np.log(tot) - log_shift
        v = v - log_norm[:,np.newaxis]

    return (v, log_norm)

def expect_log_sticks(sticks):
    """
    For stick-breaking hdp, this returns the E[log(sticks)] 
    """
    dig_sum = sp.psi(np.sum(sticks, 0))
    ElogW = sp.psi(sticks[0]) - dig_sum
    Elog1_W = sp.psi(sticks[1]) - dig_sum

    n = len(sticks[0]) + 1
    Elogsticks = np.zeros(n)
    Elogsticks[0:n-1] = ElogW
    Elogsticks[1:] = Elogsticks[1:] + np.cumsum(Elog1_W)
    return Elogsticks 

class suff_stats:
    def __init__(self, T, dim, size, mode):
        # T: top level topic number
        # dim: dimension
        # size: batch size
        self.batchsize = size
        self.var_stick = np.zeros(T) 
        self.var_x0 = np.zeros(T)
        self.var_x1 = np.zeros((T, dim))
        if mode == 'full':
            self.var_x2 = np.zeros((T, dim, dim))
        elif mode == 'diagonal':
            self.var_x2 = np.zeros((T, dim))
        elif mode == 'spherical':
            self.var_x2 = np.zeros(T)
        elif mode == 'semi-spherical':
            self.var_x2 = np.zeros(T)
        else:
            raise NoSuchModeError
class Data:
    def __init__(self):
        pass
    def next(self):
        """
        return next point
        """
        pass
    def sampele(self, n):
        """
        sample n points
        """
        pass
    def reset(self):
        """
        reset from the begining
        """
        pass

class StreamData(Data):
    def __init__(self, stream, parse_func = None):
        self.stream = stream
    def next(self):
        if not parse_func:
            return parse_func(self.stream)
        line = self.stream.readline().strip()
        x = [float(r) for r in line.split()]
        return np.array(x)
    def sample(self, n):
        sample = []
        for i in range(n):
            sample.append(self.next())
        return np.array(sample)
    def reset(self):
        self.stream.seek(0)

class ListData(Data):
    def __init__(self, X):
        self.X = np.copy(X)
        self._index = 0
    def next(self):
        if self._index >= self.X.shape[0]:
            self._index = 0
        x = self.X[self._index]
        self._index += 1
        return x
    def sample(self, n):
        s = np.random.choice(self.X.shape[0], n)
        return self.X[s,:]
    def reset(self):
        self._index = 0

class RandomGaussMixtureData(Data):
    """
    generate Gausssian mixture data
    """
    def __init__(self, weight, mean, cov):
        self.weight = weight
        self.mean = mean
        self.cov = cov
    def next(self):
        c = np.random.choice(len(self.weight), p = self.weight)
        return np.random.multivariate_normal(self.mean[c], self.cov[c])
    def sample(self, n):
        #c = np.random.choice(len(self.weight), p = self.weight, size = n)
        #smpl = map(lambda x: np.random.multivariate_normal(self.mean[x],
        #    self.cov[x]), c)
        count = np.random.multinomial(n, self.weight)
        data = np.zeros((n, self.mean.shape[1]))
        start = 0
        for i in range(len(count)):
            data[start:start+count[i],:] = \
                np.random.multivariate_normal(\
                    self.mean[i], self.cov[i], count[i])
            start = start + count[i]
        s = np.arange(n)
        np.random.shuffle(s)
        return data[s]

class online_dp:
    ''' hdp model using stick breaking'''
    def __init__(self, T, gamma, kappa, tau, total, dim, mode,
            init_mean=None, init_cov=1.0, prior_x0=None):
        """ T: top level truncation level
        gamma: first level concentration
        kappa: learning rate
        tau: slow down parameter
        total: total number of data
        dim: dimensionality of vector
        mode: covarance matrix mode
        """
        self.m_T = T # Top level truncation
        self.m_gamma = gamma # first level truncation
        self.m_total = total # total ponits

        ## for online learning
        self.m_tau = tau
        self.m_kappa = kappa
        self.m_updatect = 0 # update count

        ## each column is the top level topic's beta distribution para
        self.var_stick = np.zeros((2, T-1))
        self.var_stick[0] = 1.0
        self.var_stick[1] = self.m_gamma

        self.var_varphi = np.zeros(T)

        self.m_dim = dim # the vector dimension
        ## mode: spherical, diagonal, full
        self.mode = mode
        ## the prior of each gaussian
        if init_mean == None or init_mean.shape != (self.m_T, self.m_dim):
            print 'random init mean'
            self.m_mean = np.random.normal(0, 1, (self.m_T, self.m_dim))
        else:
            self.m_mean = init_mean
        self.m_const = np.zeros(self.m_T)
        if mode == 'full':
            self.m_precis = np.tile(\
                np.eye(self.m_dim) / init_cov, (self.m_T, 1, 1)) 
            if prior_x0 == None or prior_x0 < self.m_dim + 2:
                self.prior_x0 = self.m_dim + 2
            else:
                self.prior_x0 = prior_x0
            cov = np.tile(np.eye(self.m_dim) * init_cov, (self.m_T, 1, 1)) 

            self.prior_x2 = self.prior_x0 * cov
            self.var_x0 = np.ones(self.m_T) * self.prior_x0
            self.var_x1 = self.prior_x0 * self.m_mean
            self.var_x2 = self.prior_x0 * (self.m_mean[:,np.newaxis,:] *\
                self.m_mean[:,:,np.newaxis] + cov)
        elif mode == 'diagonal':
            self.m_precis = np.ones((self.m_T, self.m_dim)) / init_cov
            if prior_x0 == None:
                self.prior_x0 = 1
            else:
                self.prior_x0 = prior_x0
            self.prior_x2 = (self.prior_x0+2) * init_cov * np.ones(self.m_dim)
            self.var_x0 = np.ones(self.m_T) * self.prior_x0
            self.var_x1 = self.m_mean * self.prior_x0
            self.var_x2 = (self.prior_x0+2) * init_cov +\
                self.prior_x0 * (self.m_mean ** 2)
        elif mode == 'spherical':
            self.m_precis = np.ones(self.m_T) / init_cov
            if prior_x0 == None:
                self.prior_x0 = 1
            else:
                self.prior_x0 = prior_x0
            self.prior_x2 = (self.m_dim * self.prior_x0 - self.m_dim + 2) *\
                init_cov
            self.var_x0 = np.ones(self.m_T) * self.prior_x0
            self.var_x1 = self.m_mean * self.prior_x0
            self.var_x2 = self.prior_x0 * np.sum(self.m_mean ** 2 , 1) + \
                (self.m_dim * self.prior_x0 - self.m_dim + 2) * init_cov
        elif mode == 'semi-spherical':
            self.m_precis = np.ones(self.m_T) / init_cov
            # prior = (mean_x0, precis_x0)
            if prior_x0 == None:
                self.prior_x0 = (1, 100000)
            else:
                self.prior_x0 = prior_x0
            self.prior_x2 = (self.m_dim*self.prior_x0[1] + 2) * init_cov

            self.var_x0 = np.tile(self.prior_x0, (self.m_T, 1))
            self.var_x1 = self.m_mean * self.prior_x0[0]
            self.var_x2 = (self.m_dim*self.prior_x0[1] + 2) * init_cov
        else:
            raise NoSuchModeError

        self.update_par(self.var_x2, self.var_x1, self.var_x0)

    def get_cov(self):
        """return covariance matrix
        """
        cov = np.empty((self.m_T, self.m_dim, self.m_dim), dtype = 'float64')
        if self.mode == 'full':
            for t in range(self.m_T):
                cov[t] = linalg.inv(self.m_precis[t])
        elif self.mode == 'diagonal':
            for t in range(self.m_T):
                cov[t] = np.diag(1.0 / self.m_precis[t])
        elif self.mode == 'spherical':
            for t in range(self.m_T):
                cov[t] = np.eye(self.m_dim) / self.m_precis[t]
        elif self.mode == 'semi-spherical':
            for t in range(self.m_T):
                cov[t] = np.eye(self.m_dim) / self.m_precis[t]
        else:
            raise NoSuchModeError
        return cov

    def update_par(self, x2, x, r):
        """update m_mean, m_precis, m_const
        """
        #self.m_var_x = x
        if self.mode == 'full':
            mean = x / r[:, np.newaxis]
            self.m_mean = mean
            cov = x2 / r[:,np.newaxis, np.newaxis] - mean[:,:,np.newaxis] *\
                mean[:,np.newaxis,:]
            for t in range(self.m_T):
                self.m_precis[t] = linalg.inv(cov[t])
                self.m_const[t] = 0.5 * (np.log(linalg.det(self.m_precis[t])) +\
                    np.sum(sp.psi(0.5 * \
                    (self.var_x0[t] - np.arange(self.m_dim)))))
            self.m_const -= 0.5 * self.m_dim * \
                (np.log(self.var_x0 * 0.5) + \
                    1.0 / self.var_x0 + np.log(2 * np.pi))
            #self.m_var_x2 = (cov + mean[:, :, np.newaxis] * mean[:, np.newaxis, :]) * r[:, np.newaxis, np.newaxis]
        elif self.mode == 'diagonal':
            mean = x / r[:, np.newaxis]
            self.m_mean = mean
            a = 0.5 * (r + 2)
            b = 0.5 * (x2 - mean * mean * r[:, np.newaxis])
            self.m_precis = (r[:,np.newaxis] + 2) * 0.5 / b
            self.m_const = 0.5 * self.m_dim * \
                (sp.psi(a)  - 1.0 / r - np.log(2 * np.pi)) -\
                0.5 * np.sum(np.log(b), 1)
            #self.m_var_x2 = 2 * cov + r[:, np.newaxis] * mean * mean
        elif self.mode == 'spherical':
            mean = x / r[:, np.newaxis]
            self.m_mean = mean
            b = 0.5 * (x2 - np.sum(mean * mean, 1) * r)
            a = 0.5 * (self.m_dim * r - self.m_dim + 2)
            self.m_precis = a / b
            self.m_const = 0.5 * self.m_dim * (sp.psi(a) - 1.0 / r \
                - np.log(b) - np.log(2 * np.pi))
            #self.m_var_x2 = 2 * cov + r * np.sum(mean * mean, 1)
        elif self.mode == 'semi-spherical':
            mean = x / r[:,0][:, np.newaxis]
            self.m_mean = mean
            # precision of the mean
            self.m_precis = (self.m_dim*r[:, 1] + 2) / x2
            self.m_const = 0.5*self.m_dim * (
                    np.log(self.m_precis) - np.log(np.pi * 2) 
                    - 1.0 / r[:, 0])
        else:
            raise NoSuchModeError

    def par_to_natural(self, cov, mean, r):
        raise Exception("don't use this")
        x = mean * r[:, np.newaxis]
        if self.mode == 'full':
            x2 = cov + mean[:, :, np.newaxis] * mean[:, np.newaxis, :]
            x2 = x2 * r[:, np.newaxis, np.newaxis]
        elif self.mode == 'diagonal':
            x2 = 2 * cov + r[:, np.newaxis] * mean * mean
        elif self.mode == 'spherical':
            x2 = 2 * cov + r * np.sum(mean * mean, 1)
        elif self.mode == 'semi-spherical':
            x2 = (self.m_dim*r + 2) * cov
        else:
            raise NoSuchModeError
        return x2, x

    def new_init(self, c):
        """"no use"""
        raise Exception("no use")
        np.random.shuffle(c)
        self.m_mean[:] = c[0:self.m_T]

    def process_documents(self, cops, var_converge = 0.000001):
        size = 0
        for c in cops:
            size += c.shape[0]
        ss = suff_stats(self.m_T, self.m_dim, size, self.mode) 
        Elogsticks_1st = expect_log_sticks(self.var_stick) 

        score = 0.0
        for i, cop in enumerate(cops):
            cop_score = self.doc_e_step(cop, ss, Elogsticks_1st, var_converge)
            score += cop_score

        self.update_model(ss)
        return score

    def doc_e_step(self, X, ss, Elogsticks_1st, var_converge, max_iter=100):
        likelihood = 0.0
        old_likelihood = -1e100
        converge = 1.0 
        eps = 1e-100
        iter = 0
        
        Eloggauss = self.E_log_gauss(X)
        z = Eloggauss + Elogsticks_1st
        z, norm = log_normalize(z)
        z = np.exp(z)
        self.add_to_sstats(z, z, X, ss)

        return likelihood

    def add_to_sstats(self, var_phi, z, X, ss):
        ss.var_stick += np.sum(var_phi, 0)   
        ss.var_x0 += np.sum(z, 0)
        ss.var_x1 += np.sum(X[:,np.newaxis,:] * z[:,:,np.newaxis], axis = 0)
        if self.mode == 'full':
            for n in range(X.shape[0]):
                x2 = X[n,:,np.newaxis] * X[n,np.newaxis,:]
                ss.var_x2 += x2[np.newaxis,:,:] * z[n,:,np.newaxis,np.newaxis]
        elif self.mode == 'diagonal':
            x2 = X * X
            ss.var_x2 += np.sum(x2[:,np.newaxis,:] * z[:,:,np.newaxis], 0)
        elif self.mode == 'spherical':
            x2 = np.sum(X * X, 1)
            ss.var_x2 += np.sum(x2[:,np.newaxis] * z, 0)
        elif self.mode == 'semi-spherical':
            # TODO: check 0 or 1
            const = self.m_dim / (self.var_x0[:,0] * self.m_precis)
            for n in range(X.shape[0]):
                dx = X[n][np.newaxis,:] - self.m_mean
                ss.var_x2 += (np.sum(dx * dx, 1) + const) * z[n]
        else:
            raise NoSuchModeError

    def fit(self, X, size = 200, max_iter = 1000):
        #self.new_init(X)
        for i in range(max_iter):
            samples = np.array(\
                np.random.sample(size) * X.shape[0], dtype = 'int32')
            data = X[samples]
            self.process_documents([data])

    def predict(self, X):
        Elogsticks_1st = expect_log_sticks(self.var_stick) 
        res = self.E_log_gauss(X) + Elogsticks_1st
        return res.argmax(axis=1)

    def E_log_gauss(self, X):
        ds = self.diff_square(X)
        return -0.5 * ds + self.m_const[np.newaxis]

    def diff_square(self, X):
        ds = np.zeros((X.shape[0], self.m_T))
        if self.mode == 'full':
            for t in range(self.m_T):
                ds[:,t] = (distance.cdist(X, self.m_mean[t][np.newaxis], \
                    "mahalanobis", VI=self.m_precis[t]) ** 2).reshape(-1)
        elif self.mode == 'diagonal':
            for t in range(self.m_T):
                ds[:,t] = np.sum(((X - self.m_mean[t]) ** 2) *\
                    self.m_precis[t], 1)
        elif self.mode == 'spherical':
            for t in range(self.m_T):
                ds[:,t] = np.sum(((X - self.m_mean[t]) ** 2), 1)\
                    * self.m_precis[t]
        elif self.mode == 'semi-spherical':
            for t in range(self.m_T):
                ds[:,t] = np.sum(((X - self.m_mean[t]) ** 2), 1)\
                    * self.m_precis[t]
        else:
            raise NoSuchModeError
        return ds

    def update_model(self, sstats):
        # rhot will be between 0 and 1, and says how much to weight
        # the information we got from this mini-batch.

        rhot = pow(self.m_tau + self.m_updatect, -self.m_kappa)
        if rhot < rhot_bound: 
            rhot = rhot_bound
        self.m_rhot = rhot
        self.m_updatect += 1

        scale = self.m_total / sstats.batchsize

        self.var_varphi = (1.0-rhot) * self.var_varphi + \
            rhot * scale * sstats.var_stick
        ## update top level sticks 
        self.var_stick[0] = self.var_varphi[:self.m_T-1]  + 1.0
        varphi_sum = np.flipud(self.var_varphi[1:])
        self.var_stick[1] = np.flipud(np.cumsum(varphi_sum)) + self.m_gamma

        if self.mode == 'semi-spherical':
            var_x0 = (1 - rhot) * self.var_x0 + \
                rhot*(self.prior_x0 \
                + scale * sstats.var_x0[:, np.newaxis])
        else:
            var_x0 = (1 - rhot) * self.var_x0 + \
                rhot*(self.prior_x0 + scale * sstats.var_x0)
        var_x1 = (1 - rhot)* self.var_x1 + \
            rhot * scale * sstats.var_x1 # note: prior_x1 = 0
        var_x2 = (1 - rhot)* self.var_x2 + \
            rhot * (self.prior_x2 + scale * sstats.var_x2)
        self.update_par(var_x2, var_x1, var_x0)
        self.var_x0 = var_x0
        self.var_x1 = var_x1
        self.var_x2 = var_x2


    def save_model(self, output):
        model = {'sticks':self.var_stick,
                'means': self.m_mean,
                'precis':self.m_cov}
        cPickle.dump(model, output)

class Group:
    def __init__(self, alpha, size, data):
        self.m_alpha = alpha
        #v = np.zeros((2, self.m_K - 1))
        #v[0] = 1.0
        #v[1] = alpha
        #self.m_v = v
        self.m_v = None # 2 * (K - 1) array
        self.m_var_phi = None # K * T array
        self.size = size # don't need to be the same the data
        self.data = data
        self.update_timect = -10 # times of updating parameter
    def report(self):
        weight = np.exp(expect_log_sticks(self.m_v))
        print 'weight:' , weight
        print 'varphi:' , self.m_var_phi
        
class online_hdp(online_dp):
    ''' hdp model using stick breaking'''
    def __init__(self, T, K, D, alpha, gamma, kappa, tau, total, dim, mode):
        """
        gamma: first level concentration
        alpha: second level concentration
        T: top level truncation level
        K: second level truncation level
        D: number of documents in the corpus
        kappa: learning rate
        tau: slow down parameter
        """
        online_dp.__init__(self, T, gamma, kappa, tau, total, dim, mode)
        self.m_K = K # second level truncation
        self.m_alpha = alpha # second level concentration

    def process_groups(self, groups):
        ## should remove batchsize from suff_stats for 
        ## batch_size = m_rel.sum()
        ## TODO fix batch_size
        size = 1000
        batch_size = 500
        #for c in groups:
            #size += c.shape[0]
        ss = suff_stats(self.m_T, self.m_dim, size, self.mode) 
        Elogsticks_1st = expect_log_sticks(self.var_stick) 

        score = 0.0
        for group in groups:
            if group.update_timect <= 0:
                ## first time for this group
                #debug('init group')
                score += self.init_group(group, ss, Elogsticks_1st, batch_size)
            else:
                #score += self.process_group(group, ss, Elogsticks_1st, batch_size)
                score += self.c_process_group(\
                    group, ss, Elogsticks_1st, batch_size)
        self.update_model(ss)
        return score

    def init_group(self, group, ss, Elogsticks_1st, batch_size, \
            var_converge = 0.000001, max_iter=100):
        ## very similar to the hdp equations
        v = np.zeros((2, self.m_K-1)) 
        v[0] = 1.0
        v[1] = self.m_alpha

        # The following line is of no use.
        Elogsticks_2nd = expect_log_sticks(v)

        # back to the uniform
        X = group.data.sample(batch_size)
        phi = np.ones((X.shape[0], self.m_K)) / self.m_K

        likelihood = 0.0
        old_likelihood = -1e100
        converge = 1.0 
        eps = 1e-100
        iter = 0
        
        Eloggauss = self.E_log_gauss(X)
        # del var_phi
        #var_phi = None
        while iter < 10 or (iter < max_iter \
            and (converge <= 0.0 or converge > var_converge)):
        #while iter < max_iter:
            ### update variational parameters
            # var_phi 
            if iter < 5:
                var_phi = np.dot(phi.T, Eloggauss)
                (log_var_phi, log_norm) = log_normalize(var_phi)
                var_phi = np.exp(log_var_phi)
            else:
                var_phi = np.dot(phi.T,  Eloggauss) + Elogsticks_1st
                (log_var_phi, log_norm) = log_normalize(var_phi)
                var_phi = np.exp(log_var_phi)
            
            # phi
            if iter < 5:
                phi = np.dot(Eloggauss, var_phi.T)
                (log_phi, log_norm) = log_normalize(phi)
                phi = np.exp(log_phi)
            else:
                phi = np.dot(Eloggauss, var_phi.T) + Elogsticks_2nd
                (log_phi, log_norm) = log_normalize(phi)
                phi = np.exp(log_phi)

            # v
            v[0] = 1.0 + np.sum(phi[:,:self.m_K-1], 0)
            phi_cum = np.flipud(np.sum(phi[:,1:], 0))
            v[1] = self.m_alpha + np.flipud(np.cumsum(phi_cum))
            Elogsticks_2nd = expect_log_sticks(v)
            #debug(np.exp(Elogsticks_2nd))

            ## TODO: likelihood need complete
            likelihood = 0.0
            # compute likelihood
            # var_phi part/ C in john's notation
            likelihood += np.sum((Elogsticks_1st - log_var_phi) * var_phi)

            # v part/ v in john's notation, john's beta is alpha here
            log_alpha = np.log(self.m_alpha)
            likelihood += (self.m_K-1) * log_alpha
            dig_sum = sp.psi(np.sum(v, 0))
            likelihood += np.sum(\
                (np.array([1.0, self.m_alpha])[:,np.newaxis]-v) *\
                    (sp.psi(v)-dig_sum))
            likelihood -= np.sum(sp.gammaln(np.sum(v, 0))) \
                - np.sum(sp.gammaln(v))

            # Z part 
            likelihood += np.sum((Elogsticks_2nd - log_phi) * phi)

            # X part, the data part
            likelihood += np.sum(phi.T * np.dot(var_phi, Eloggauss.T))

            #debug(likelihood, old_likelihood)

            converge = (likelihood - old_likelihood)/abs(old_likelihood)
            old_likelihood = likelihood

            if converge < -0.000001:
                print "warning, likelihood is decreasing!"
            
            iter += 1
        #debug(iter)
        # update the suff_stat ss 
        group.m_v = v
        group.m_var_phi = var_phi + np.finfo(np.float32).eps
        group.update_timect += 1
        z = np.dot(phi, var_phi) 
        self.add_to_sstats(var_phi, z, X, ss)
        return likelihood

    def predict(self, X, group = None):
        Elogsticks_1st = expect_log_sticks(self.var_stick) 
        if group is None:
            res = self.E_log_gauss(X) + Elogsticks_1st
            return res.argmax(axis=1)

        # The following line is of no use.
        Elogsticks_2nd = expect_log_sticks(group.m_v)
        Esticks = np.exp(Elogsticks_2nd)
        weight = np.sum(Esticks[:,np.newaxis] * group.m_var_phi, axis = 0)
        logweight = np.log(weight)
        logpost = self.E_log_gauss(X) + logweight[np.newaxis,:]
        return logpost.argmax(axis=1)

    def process_group(self, group, ss, Elogsticks_1st, batch_size):
        X = group.data.sample(batch_size)
        v = group.m_v.copy()
        var_phi = group.m_var_phi

        # The following line is of no use.
        Elogsticks_2nd = expect_log_sticks(v)
        Eloggauss = self.E_log_gauss(X)

        phi = np.dot(Eloggauss, var_phi.T) + Elogsticks_2nd
        (log_phi, log_norm) = log_normalize(phi)
        phi = np.exp(log_phi)

        var_phi = np.dot(phi.T,  Eloggauss) + Elogsticks_1st
        (log_var_phi, log_norm) = log_normalize(var_phi)
        var_phi = np.exp(log_var_phi)
        ## TODO
        rhot = pow(self.m_tau + group.update_timect, -self.m_kappa)
        group.update_timect += 1
        scale = float(group.size) / batch_size

        ## update group parameter m_v
        v[0] = 1.0 + scale * np.sum(phi[:,:self.m_K-1], 0)
        phi_cum = np.flipud(np.sum(phi[:,1:], 0))
        v[1] = self.m_alpha + scale * np.flipud(np.cumsum(phi_cum))
        group.m_v = (1 - rhot) * group.m_v + rhot * v

        ## update group parameter m_var_phi
        ## notice: the natual parameter is log(var_phi)
        log_m_var_phi = np.log(group.m_var_phi)
        log_m_var_phi = (1 - rhot) * log_m_var_phi + rhot * log_var_phi
        group.m_var_phi = np.exp(log_m_var_phi)

        # compute likelihood
        # var_phi part/ C in john's notation
        likelihood = 0.0
        likelihood += np.sum((Elogsticks_1st - log_var_phi) * var_phi)

        # v part/ v in john's notation, john's beta is alpha here
        log_alpha = np.log(self.m_alpha)
        likelihood += (self.m_K-1) * log_alpha
        dig_sum = sp.psi(np.sum(v, 0))
        likelihood += np.sum((np.array([1.0, self.m_alpha])[:,np.newaxis]-v)\
            * (sp.psi(v)-dig_sum))
        likelihood -= np.sum(sp.gammaln(np.sum(v, 0))) - np.sum(sp.gammaln(v))

        # Z part 
        likelihood += np.sum((Elogsticks_2nd - log_phi) * phi)

        # X part, the data part
        likelihood += np.sum(phi.T * np.dot(var_phi, Eloggauss.T))

        #debug(likelihood, old_likelihood)

        #debug(iter)    
        # update the suff_stat ss 
        z = np.dot(phi, var_phi) 
        self.add_to_sstats(var_phi, z, X, ss)
        return likelihood

    def c_process_group(self, group, ss, Elogsticks_1st, batch_size,\
            var_converge = 0.000001, max_iter=20):
        X = group.data.sample(batch_size)
        phi = np.ones((X.shape[0], self.m_K)) / self.m_K
        v = group.m_v.copy()
        var_phi = group.m_var_phi.copy()
        Elogsticks_2nd = expect_log_sticks(v)
        Eloggauss = self.E_log_gauss(X)

        likelihood = 0.0
        old_likelihood = -1e100
        converge = 1.0 
        eps = 1e-100
        iter = 0
        while iter < 3 or (iter < max_iter \
                and (converge <= 0.0 or converge > var_converge)):
            if iter < 5:
                var_phi = np.dot(phi.T, Eloggauss)
                (log_var_phi, log_norm) = log_normalize(var_phi)
                var_phi = np.exp(log_var_phi)
            else:
                var_phi = np.dot(phi.T,  Eloggauss) + Elogsticks_1st
                (log_var_phi, log_norm) = log_normalize(var_phi)
                var_phi = np.exp(log_var_phi)
            
            # phi
            if iter < 5:
                phi = np.dot(Eloggauss, var_phi.T)
                (log_phi, log_norm) = log_normalize(phi)
                phi = np.exp(log_phi)
            else:
                phi = np.dot(Eloggauss, var_phi.T) + Elogsticks_2nd
                (log_phi, log_norm) = log_normalize(phi)
                phi = np.exp(log_phi)

            # v
            v[0] = 1.0 + np.sum(phi[:,:self.m_K-1], 0)
            phi_cum = np.flipud(np.sum(phi[:,1:], 0))
            v[1] = self.m_alpha + np.flipud(np.cumsum(phi_cum))
            Elogsticks_2nd = expect_log_sticks(v)
            #debug(np.exp(Elogsticks_2nd))

            ## TODO: likelihood need complete
            likelihood = 0.0
            # compute likelihood
            # var_phi part/ C in john's notation
            likelihood += np.sum((Elogsticks_1st - log_var_phi) * var_phi)

            # v part/ v in john's notation, john's beta is alpha here
            log_alpha = np.log(self.m_alpha)
            likelihood += (self.m_K-1) * log_alpha
            dig_sum = sp.psi(np.sum(v, 0))
            likelihood += np.sum(\
                (np.array([1.0, self.m_alpha])[:,np.newaxis]-v) *\
                    (sp.psi(v)-dig_sum))
            likelihood -= np.sum(sp.gammaln(np.sum(v, 0))) \
                - np.sum(sp.gammaln(v))

            # Z part 
            likelihood += np.sum((Elogsticks_2nd - log_phi) * phi)

            # X part, the data part
            likelihood += np.sum(phi.T * np.dot(var_phi, Eloggauss.T))

            #debug(likelihood, old_likelihood)

            converge = (likelihood - old_likelihood)/abs(old_likelihood)
            old_likelihood = likelihood

            if converge < -0.000001:
                print "warning, likelihood is decreasing!"
            
            iter += 1

        rhot = pow(self.m_tau + group.update_timect, -self.m_kappa)
        group.update_timect += 1
        scale = float(group.size) / batch_size

        ## update group parameter m_v
        v[0] = 1.0 + scale * np.sum(phi[:,:self.m_K-1], 0)
        phi_cum = np.flipud(np.sum(phi[:,1:], 0))
        v[1] = self.m_alpha + scale * np.flipud(np.cumsum(phi_cum))
        group.m_v = (1 - rhot) * group.m_v + rhot * v

        ## update group parameter m_var_phi
        ## notice: the natual parameter is log(var_phi)
        log_m_var_phi = np.log(group.m_var_phi)
        log_m_var_phi = (1 - rhot) * log_m_var_phi + rhot * log_var_phi
        group.m_var_phi = np.exp(log_m_var_phi)

        # compute likelihood
        # var_phi part/ C in john's notation
        likelihood = 0.0
        likelihood += np.sum((Elogsticks_1st - log_var_phi) * var_phi)

        # v part/ v in john's notation, john's beta is alpha here
        log_alpha = np.log(self.m_alpha)
        likelihood += (self.m_K-1) * log_alpha
        dig_sum = sp.psi(np.sum(v, 0))
        likelihood += np.sum((np.array([1.0, self.m_alpha])[:,np.newaxis]-v) \
            * (sp.psi(v)-dig_sum))
        likelihood -= np.sum(sp.gammaln(np.sum(v, 0))) - np.sum(sp.gammaln(v))

        # Z part 
        likelihood += np.sum((Elogsticks_2nd - log_phi) * phi)

        # X part, the data part
        likelihood += np.sum(phi.T * np.dot(var_phi, Eloggauss.T))

        #debug(likelihood, old_likelihood)

        #debug(iter)    
        # update the suff_stat ss 
        z = np.dot(phi, var_phi) 
        self.add_to_sstats(var_phi, z, X, ss)
        return likelihood

    def doc_e_step(self, X, ss, Elogsticks_1st, var_converge, max_iter=100):
        #raise Exception("should use process_group instead")
        """
        e step for a single corps
        """

        ## very similar to the hdp equations
        v = np.zeros((2, self.m_K-1))  
        v[0] = 1.0
        v[1] = self.m_alpha

        # The following line is of no use.
        Elogsticks_2nd = expect_log_sticks(v)

        # back to the uniform
        phi = np.ones((X.shape[0], self.m_K)) / self.m_K

        likelihood = 0.0
        old_likelihood = -1e100
        converge = 1.0 
        eps = 1e-100
        iter = 0
        
        Eloggauss = self.E_log_gauss(X)

        while iter < 10 or (iter < max_iter \
                and (converge <= 0.0 or converge > var_converge)):
        #while iter < max_iter:
            ### update variational parameters
            # var_phi 
            if iter < 5:
                var_phi = np.dot(phi.T, Eloggauss)
                (log_var_phi, log_norm) = log_normalize(var_phi)
                var_phi = np.exp(log_var_phi)
            else:
                var_phi = np.dot(phi.T,  Eloggauss) + Elogsticks_1st
                (log_var_phi, log_norm) = log_normalize(var_phi)
                var_phi = np.exp(log_var_phi)
            
            # phi
            if iter < 5:
                phi = np.dot(Eloggauss, var_phi.T)
                (log_phi, log_norm) = log_normalize(phi)
                phi = np.exp(log_phi)
            else:
                phi = np.dot(Eloggauss, var_phi.T) + Elogsticks_2nd
                (log_phi, log_norm) = log_normalize(phi)
                phi = np.exp(log_phi)

            # v
            v[0] = 1.0 + np.sum(phi[:,:self.m_K-1], 0)
            phi_cum = np.flipud(np.sum(phi[:,1:], 0))
            v[1] = self.m_alpha + np.flipud(np.cumsum(phi_cum))
            Elogsticks_2nd = expect_log_sticks(v)
            #debug(np.exp(Elogsticks_2nd))

            ## TODO: likelihood need complete
            likelihood = 0.0
            # compute likelihood
            # var_phi part/ C in john's notation
            likelihood += np.sum((Elogsticks_1st - log_var_phi) * var_phi)

            # v part/ v in john's notation, john's beta is alpha here
            log_alpha = np.log(self.m_alpha)
            likelihood += (self.m_K-1) * log_alpha
            dig_sum = sp.psi(np.sum(v, 0))
            likelihood += np.sum(\
                (np.array([1.0, self.m_alpha])[:,np.newaxis]-v) *\
                    (sp.psi(v)-dig_sum))
            likelihood -= np.sum(sp.gammaln(np.sum(v, 0))) \
                - np.sum(sp.gammaln(v))

            # Z part 
            likelihood += np.sum((Elogsticks_2nd - log_phi) * phi)

            # X part, the data part
            likelihood += np.sum(phi.T * np.dot(var_phi, Eloggauss.T))

            #debug(likelihood, old_likelihood)

            converge = (likelihood - old_likelihood)/abs(old_likelihood)
            old_likelihood = likelihood

            if converge < -0.000001:
                print "warning, likelihood is decreasing!"
            
            iter += 1
        #debug(iter)    
        # update the suff_stat ss 
        z = np.dot(phi, var_phi) 
        self.add_to_sstats(var_phi, z, X, ss)
        return likelihood
