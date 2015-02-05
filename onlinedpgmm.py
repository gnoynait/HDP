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
    """For stick-breaking hdp, this returns the E[log(sticks)] 
    """
    dig_sum = sp.psi(np.sum(sticks, 0))
    ElogW = sp.psi(sticks[0]) - dig_sum
    Elog1_W = sp.psi(sticks[1]) - dig_sum

    n = len(sticks[0]) + 1
    Elogsticks = np.zeros(n)
    Elogsticks[0:n-1] = ElogW
    Elogsticks[1:] = Elogsticks[1:] + np.cumsum(Elog1_W)
    return Elogsticks 

class SuffStats:
    """sufficient statistics
    """
    def __init__(self, T, dim, mode):
        # T: top level topic number
        # dim: dimension
        # size: batch size
        self.batchsize = 0
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


class FileData:
    def __init__(self, fname, parser = None):
        self.dfile = open(fname)
        self.parser = parser
        self.count = 0
    def size(self):
        self.dfile.seek(0)
        s = 0
        line = self.dfile.readline()
        while line != '':
            s += 1
            line = self.dfile.readline()
        return s
            
    def sample(self, n):
        samples = []
        for i in range(n):
            line = self.dfile.readline()
            if not line:
                self.dfile.seek(0)
                line = self.dfile.readline()
            if self.parser is None:
                x = [float(r) for r in line.split()]
            else:
                x = self.parser(line)
            samples.append(x)
        return np.array(samples)
    def reset(self):
        self.dfile.seek(0)
    def next_n_record(self, n):
        topics = []
        X = []
        for line in self.dfile:
            if line is None or n == 0:
                break
            n -= 1
            record = line.split()
            topics.append(record[0])
            #titles.append(record[1])
            X.append([float(a) for a in record[1:]])
        return topics, np.array(X)


class ListData:
    def __init__(self, X):
        self.X = np.copy(X)
    def sample(self, n):
        s = np.random.choice(self.X.shape[0], n)
        return self.X[s,:]

class RandomGaussMixtureData:
    """generate Gausssian mixture data
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
    def save_data(self, fname, n):
        """save n samples to file
        """
        X = self.sample(n)
        np.savetxt(fname, X)
        

class OnlineDP:
    """Online DP model"""
    def __init__(self, T, gamma, kappa, tau, total, dim, mode):
            #init_mean=None, init_cov=1.0, prior_x0=None):
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
        self.init_par()

    def init_par(self, init_mean=None, init_cov=1, prior_x0=None):
        ## the prior of each gaussian
        if init_mean is None or init_mean.shape != (self.m_T, self.m_dim):
            print 'random init mean'
            self.m_mean = np.random.normal(0, 1, (self.m_T, self.m_dim))
        else:
            np.random.shuffle(init_mean)
            self.m_mean = init_mean[0:self.m_T]
        self.m_const = np.zeros(self.m_T)
        if self.mode == 'full':
            self.m_precis = np.tile(\
                np.eye(self.m_dim) / init_cov, (self.m_T, 1, 1)) 
            if prior_x0 is None or prior_x0 < self.m_dim + 2:
                self.prior_x0 = self.m_dim + 2
            else:
                self.prior_x0 = prior_x0
            cov = np.tile(np.eye(self.m_dim) * init_cov, (self.m_T, 1, 1)) 

            self.prior_x2 = self.prior_x0 * cov
            self.var_x0 = np.ones(self.m_T) * self.prior_x0
            self.var_x1 = self.prior_x0 * self.m_mean
            self.var_x2 = self.prior_x0 * (self.m_mean[:,np.newaxis,:] *\
                self.m_mean[:,:,np.newaxis] + cov)
        elif self.mode == 'diagonal':
            self.m_precis = np.ones((self.m_T, self.m_dim)) / init_cov
            if prior_x0 is None:
                self.prior_x0 = 1
            else:
                self.prior_x0 = prior_x0
            self.prior_x2 = (self.prior_x0+2) * init_cov * np.ones(self.m_dim)
            self.var_x0 = np.ones(self.m_T) * self.prior_x0
            self.var_x1 = self.m_mean * self.prior_x0
            self.var_x2 = (self.prior_x0+2) * init_cov +\
                self.prior_x0 * (self.m_mean ** 2)
        elif self.mode == 'spherical':
            self.m_precis = np.ones(self.m_T) / init_cov
            if prior_x0 is None:
                self.prior_x0 = 1
            else:
                self.prior_x0 = prior_x0
            self.prior_x2 = (self.m_dim * self.prior_x0 - self.m_dim + 2) *\
                init_cov
            self.var_x0 = np.ones(self.m_T) * self.prior_x0
            self.var_x1 = self.m_mean * self.prior_x0
            self.var_x2 = self.prior_x0 * np.sum(self.m_mean ** 2 , 1) + \
                (self.m_dim * self.prior_x0 - self.m_dim + 2) * init_cov
        elif self.mode == 'semi-spherical':
            self.m_precis = np.ones(self.m_T) / init_cov
            # prior = (mean_x0, precis_x0)
            if prior_x0 is None:
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

    def process_documents(self, cops, var_converge = 0.000001):
        ss = SuffStats(self.m_T, self.m_dim, self.mode) 
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
        # varphi equals to z
        self.add_to_sstats(z, z, X, ss)

        return likelihood

    def add_to_sstats(self, varphi, z, X, ss):
        ss.batchsize += z.sum()
        ss.var_stick += np.sum(varphi, 0)   
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
    """Data group
    """
    def __init__(self, alpha, K, T, size, batchsize, data, \
            coldstart=False, maxiter=100, online=True):
        self.m_alpha = alpha
        self.m_K = K # second level
        self.m_T = T # first level
        v = np.zeros((2, self.m_K - 1))
        v[0] = 1.0
        v[1] = alpha
        self.m_v = v
        self.m_varphi = np.zeros((K, T)) # K * T array
        self.size = size # don't need to be the same the data
        self.batchsize = batchsize
        self.data = data
        self.update_timect = 1 # times of updating parameter
        self.coldstart = coldstart
        self.maxiter = maxiter
        self.online = online
    def sample(self):
        return self.data.sample(self.batchsize)
    def report(self):
        weight = np.exp(expect_log_sticks(self.m_v))
        print 'weight:' , weight
        print 'varphi:' , self.m_varphi
        
class OnlineHDP(OnlineDP):
    """Online HDP Model"""
    def __init__(self, T, K, alpha, gamma, kappa, tau, total, dim, mode):
        """
        gamma: first level concentration
        alpha: second level concentration
        T: top level truncation level
        K: second level truncation level
        total: number of documents in the corpus
        kappa: learning rate
        tau: slow down parameter
        """
        OnlineDP.__init__(self, T, gamma, kappa, tau, total, dim, mode)
        self.m_K = K # second level truncation
        self.m_alpha = alpha # second level concentration

    def process_groups(self, groups, fast=True):
        ss = SuffStats(self.m_T, self.m_dim, self.mode) 
        Elogsticks_1st = expect_log_sticks(self.var_stick) 

        score = 0.0
        for group in groups:
            if fast:
                score += self.fast_process_group(group, ss, Elogsticks_1st)
            else:
                score += self.process_group(group, ss, Elogsticks_1st)
        self.update_model(ss)
        return score

    def process_group(self, group, ss, Elogsticks_1st,\
            var_converge=0.000001):
        X = group.sample()

        if group.coldstart:
            v = np.zeros((2, group.m_K-1))
            v[0] = 1.0
            v[1] = group.m_alpha
        else:
            v = group.m_v.copy()
            varphi = group.m_varphi.copy()

        Elogsticks_2nd = expect_log_sticks(v)
        Eloggauss = self.E_log_gauss(X)

        # bug fix: this is no use
        phi = np.ones((X.shape[0], self.m_K)) / self.m_K

        likelihood = 0.0
        old_likelihood = -1e100
        converge = 1.0 
        eps = 1e-100
        iter = 0
        while iter < group.maxiter and (converge <= 0.0 or converge > var_converge):
            # varphi
            varphi = np.dot(phi.T,  Eloggauss) + Elogsticks_1st
            (log_varphi, log_norm) = log_normalize(varphi)
            varphi = np.exp(log_varphi)
            # phi
            phi = np.dot(Eloggauss, varphi.T) + Elogsticks_2nd
            (log_phi, log_norm) = log_normalize(phi)
            phi = np.exp(log_phi)
            # v
            v[0] = 1.0 + np.sum(phi[:,:self.m_K-1], 0)
            phi_cum = np.flipud(np.sum(phi[:,1:], 0))
            v[1] = self.m_alpha + np.flipud(np.cumsum(phi_cum))
            Elogsticks_2nd = expect_log_sticks(v)

            ## TODO: likelihood need complete
            likelihood = 0.0
            # compute likelihood
            # varphi part/ C in john's notation
            likelihood += np.sum((Elogsticks_1st - log_varphi) * varphi)

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
            likelihood += np.sum(phi.T * np.dot(varphi, Eloggauss.T))

            converge = (likelihood - old_likelihood)/abs(old_likelihood)
            old_likelihood = likelihood

            if converge < -0.000001:
                print "warning, likelihood is decreasing!"
            
            iter += 1

        if not group.online:
            group.m_v = v
            group.m_varphi = varphi
        else:
            rhot = pow(self.m_tau + group.update_timect, -self.m_kappa)
            scale = float(group.size) / group.batchsize

            ## update group parameter m_v
            v[0] = 1.0 + scale * np.sum(phi[:,:self.m_K-1], 0)
            phi_cum = np.flipud(np.sum(phi[:,1:], 0))
            v[1] = self.m_alpha + scale * np.flipud(np.cumsum(phi_cum))
            group.m_v = (1 - rhot) * group.m_v + rhot * v
            
            ## TODO: which version is right??
            ## update group parameter m_varphi
            ## notice: the natual parameter is log(varphi)
            """
            eps = 1.0e-100
            log_m_varphi = np.log(group.m_varphi + eps)
            log_m_varphi = (1 - rhot) * log_m_varphi + rhot * log_varphi
            group.m_varphi = np.exp(log_m_varphi)
            """
            group.m_varphi = (1 - rhot) * group.m_varphi + rhot * varphi

        group.update_timect += 1
        # compute likelihood
        # varphi part/ C in john's notation
        likelihood = 0.0
        likelihood += np.sum((Elogsticks_1st - log_varphi) * varphi)

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
        likelihood += np.sum(phi.T * np.dot(varphi, Eloggauss.T))
        # update the suff_stat ss 
        z = np.dot(phi, varphi) 
        self.add_to_sstats(varphi, z, X, ss)
        return likelihood

    def fast_process_group(self, group, ss, Elogsticks_1st):
        X = group.sample()

        v = group.m_v.copy()
        varphi = group.m_varphi.copy()

        Elogsticks_2nd = expect_log_sticks(v)
        Eloggauss = self.E_log_gauss(X)

        #phi = np.ones((X.shape[0], self.m_K)) / self.m_K

        likelihood = 0.0
        old_likelihood = -1e100
        converge = 1.0 
        eps = 1e-100
        # phi
        phi = np.dot(Eloggauss, varphi.T) + Elogsticks_2nd
        (log_phi, log_norm) = log_normalize(phi)
        phi = np.exp(log_phi)
        # varphi
        varphi = np.dot(phi.T,  Eloggauss) + Elogsticks_1st
        (log_varphi, log_norm) = log_normalize(varphi)
        varphi = np.exp(log_varphi)
        # v
        v[0] = 1.0 + np.sum(phi[:,:self.m_K-1], 0)
        phi_cum = np.flipud(np.sum(phi[:,1:], 0))
        v[1] = self.m_alpha + np.flipud(np.cumsum(phi_cum))
        Elogsticks_2nd = expect_log_sticks(v)

        ## TODO: likelihood need complete
        likelihood = 0.0
        # compute likelihood
        # varphi part/ C in john's notation
        likelihood += np.sum((Elogsticks_1st - log_varphi) * varphi)

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
        likelihood += np.sum(phi.T * np.dot(varphi, Eloggauss.T))

        rhot = pow(self.m_tau + group.update_timect, -self.m_kappa)
        scale = float(group.size) / group.batchsize

        ## update group parameter m_v
        v[0] = 1.0 + scale * np.sum(phi[:,:self.m_K-1], 0)
        phi_cum = np.flipud(np.sum(phi[:,1:], 0))
        v[1] = self.m_alpha + scale * np.flipud(np.cumsum(phi_cum))
        group.m_v = (1 - rhot) * group.m_v + rhot * v
        
        ## TODO: which version is right??
        ## update group parameter m_varphi
        ## notice: the natual parameter is log(varphi)
        """
        eps = 1.0e-100
        log_m_varphi = np.log(group.m_varphi + eps)
        log_m_varphi = (1 - rhot) * log_m_varphi + rhot * log_varphi
        group.m_varphi = np.exp(log_m_varphi)
        """
        group.m_varphi = (1 - rhot) * group.m_varphi + rhot * varphi

        group.update_timect += 1
        # compute likelihood
        # varphi part/ C in john's notation
        likelihood = 0.0
        likelihood += np.sum((Elogsticks_1st - log_varphi) * varphi)

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
        likelihood += np.sum(phi.T * np.dot(varphi, Eloggauss.T))
        # update the suff_stat ss 
        z = np.dot(phi, varphi) 
        self.add_to_sstats(varphi, z, X, ss)
        return likelihood

    def doc_e_step(self, X, ss, Elogsticks_1st, var_converge, max_iter=100):
        #raise Exception("should use process_group instead")
        """ called from the process_documents()
        e step for a single corps
        used when we don't care about group level parameters
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
            # varphi 
            if iter < 5:
                varphi = np.dot(phi.T, Eloggauss)
                (log_varphi, log_norm) = log_normalize(varphi)
                varphi = np.exp(log_varphi)
            else:
                varphi = np.dot(phi.T,  Eloggauss) + Elogsticks_1st
                (log_varphi, log_norm) = log_normalize(varphi)
                varphi = np.exp(log_varphi)
            
            # phi
            if iter < 5:
                phi = np.dot(Eloggauss, varphi.T)
                (log_phi, log_norm) = log_normalize(phi)
                phi = np.exp(log_phi)
            else:
                phi = np.dot(Eloggauss, varphi.T) + Elogsticks_2nd
                (log_phi, log_norm) = log_normalize(phi)
                phi = np.exp(log_phi)

            # v
            v[0] = 1.0 + np.sum(phi[:,:self.m_K-1], 0)
            phi_cum = np.flipud(np.sum(phi[:,1:], 0))
            v[1] = self.m_alpha + np.flipud(np.cumsum(phi_cum))
            Elogsticks_2nd = expect_log_sticks(v)

            ## TODO: likelihood need complete
            likelihood = 0.0
            # compute likelihood
            # varphi part/ C in john's notation
            likelihood += np.sum((Elogsticks_1st - log_varphi) * varphi)

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
            likelihood += np.sum(phi.T * np.dot(varphi, Eloggauss.T))

            converge = (likelihood - old_likelihood)/abs(old_likelihood)
            old_likelihood = likelihood

            if converge < -0.000001:
                print "warning, likelihood is decreasing!"

            iter += 1
        # update the suff_stat ss 
        z = np.dot(phi, varphi) 
        self.add_to_sstats(varphi, z, X, ss)
        return likelihood

    def predict(self, X, group=None, trunk=0):
        Elogsticks_1st = expect_log_sticks(self.var_stick) 
        if group is None:
            res = self.E_log_gauss(X) + Elogsticks_1st
            return res.argmax(axis=1)

        Elogsticks_2nd = expect_log_sticks(group.m_v)
        Esticks = np.exp(Elogsticks_2nd)
        weight = np.sum(Esticks[:,np.newaxis] * group.m_varphi, axis = 0)
        if trunk > 0:
            weight[weight.argsort()[:weight.size-trunk]] = 0.0
        epsilon = 1.0e-100
        logweight = np.log(weight + epsilon)
        logpost = self.E_log_gauss(X) + logweight[np.newaxis,:]
        return logpost.argmax(axis=1)

