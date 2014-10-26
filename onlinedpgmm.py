import numpy as np
import scipy.special as sp
from scipy import linalg
import os, sys, math, time
import utils
from corpus import document, corpus
from itertools import izip
import random
import cPickle
from sklearn import cluster
from scipy.special import digamma as _digamma, gammaln as _gammaln
import time
import sys

#random_seed = 999931111
random_seed = int(time.time())
np.random.seed(random_seed)
random.seed(random_seed)

rhot_bound = 0.0

def debug(*W):
    for w in W:
        sys.stderr.write(str(w) + '\n')
    sys.stderr.write('-' * 75 + '\n')

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

def digamma(x):
    return _digamma(x + np.finfo(np.float32).eps)

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
        self.m_batchsize = size
        self.m_var_sticks_ss = np.zeros(T) 
        self.m_var_res = np.zeros(T)
        self.m_var_x = np.zeros((T, dim))
        if mode == 'full':
            self.m_var_x2 = np.zeros((T, dim, dim))
        elif mode == 'diagonal':
            self.m_var_x2 = np.zeros((T, dim))
        elif mode == 'spherical':
            self.m_var_x2 = np.zeros(T)
        else:
            print 'unknow mode'
            sys.exit()

class online_dp:
    ''' hdp model using stick breaking'''
    def __init__(self, T, gamma, kappa, tau, total, dim, mode):
        """
        gamma: first level concentration
        T: top level truncation level
        kappa: learning rate
        tau: slow down parameter
        total: total number of data
        """
        self.m_T = T # Top level truncation
        self.m_gamma = gamma # first level truncation
        self.m_total = total # total ponits

        ## each column is the top level topic's beta distribution para
        self.m_var_sticks = np.zeros((2, T-1))
        self.m_var_sticks[0] = 1.0
        self.m_var_sticks[1] = self.m_gamma

        self.m_varphi_ss = np.zeros(T)

        self.m_dim = dim # the vector dimension
        ## mode: spherical, diagonal, full
        self.mode = mode
        ## the prior of each gaussian
        self.m_rel0 = np.ones(self.m_T) * (self.m_dim + 2)
        self.m_var_x0 = np.zeros((self.m_T, self.m_dim))
        self.m_means = np.random.normal(0, 1, (self.m_T, self.m_dim))
        self.m_rel = np.ones(self.m_T) * self.m_rel0
        if mode == 'full':
            #TODO check this
            self.m_var_x20 = np.tile(np.eye(self.m_dim)*0.5, (self.m_T, 1, 1)) \
                * self.m_rel0[:, np.newaxis, np.newaxis]
            self.m_cov = self.m_var_x20.copy()
            self.m_precis = np.tile(np.eye(self.m_dim)*2, (self.m_T, 1, 1)) \
                * self.m_rel0[:, np.newaxis, np.newaxis]
            self.m_log_det_precis = np.ones((self.m_T)) * self.m_dim * 2
        elif mode == 'diagonal':
            self.m_var_x20 = np.ones((self.m_T, self.m_dim)) * 0.5
            self.m_cov = self.m_var_x20.copy()
        elif mode == 'spherical':
            self.m_var_x20 = np.ones(self.m_T) * 0.5
            self.m_cov = self.m_var_x20.copy()
        else:
            print 'unkown mode'
            sys.exit()

        ## for online learning
        self.m_tau = tau + 1
        self.m_kappa = kappa
        self.m_updatect = 0 
    def get_cov(self):
        cov = np.empty((self.m_T, self.m_dim, self.m_dim), dtype = 'float64')
        for t in range(self.m_T):
            if self.mode == 'full':
                cov[t] = self.m_cov[t]
            elif self.mode == 'diagnal':
                cov[t] = np.diag(self.m_cov[t] / (0.5 * self.m_rel[t] + 1))
            elif self.mode == 'spherical':
                cov[t] = np.eye(self.m_dim) * (self.m_cov[t] / (0.5 * (self.m_dim * self.m_rel[t] - self.m_dim + 2)))
        return cov

    def natual_to_par(self, x2, x, r):
        mean = x / r[:, np.newaxis]
        if self.mode == 'full':
            cov = x2 / r[:,np.newaxis, np.newaxis] - mean[:,:,np.newaxis] * mean[:,np.newaxis,:]
        elif self.mode == 'diagonal':
            cov = 0.5 * (x2 - mean * mean * r[:, np.newaxis])
        elif self.mode == 'spherical':
            cov = 0.5 * (x2 - np.sum(mean * mean, 1) * r)
        else:
            print 'unkown'
        return cov, mean

    def par_to_natual(self, cov, mean, r):
        x = mean * r[:, np.newaxis]
        if self.mode == 'full':
            x2 = cov + mean[:, :, np.newaxis] * mean[:, np.newaxis, :]
            x2 = x2 * r[:, np.newaxis, np.newaxis]
        elif self.mode == 'diagonal':
            x2 = 2 * cov + r[:, np.newaxis] * mean * mean
        elif self.mode == 'spherical':
            x2 = 2 * cov + r * np.sum(mean * mean, 1)
        return x2, x
        
    def new_init(self, c):
        np.random.shuffle(c)
        self.m_means[:] = c[0:self.m_T]

    def process_documents(self, cops, var_converge = 0.000001):
        size = 0
        for c in cops:
            size += c.shape[0]
        ss = suff_stats(self.m_T, self.m_dim, size, self.mode) 
        Elogsticks_1st = expect_log_sticks(self.m_var_sticks) 

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
        ss.m_var_sticks_ss += np.sum(var_phi, 0)   
        ss.m_var_res += np.sum(z, axis = 0)
        ss.m_var_x += np.sum(X[:,np.newaxis,:] * z[:,:,np.newaxis], axis = 0)
        if self.mode == 'full':
            x2 = X[:,:,np.newaxis] * X[:,np.newaxis,:]
            ss.m_var_x2 += np.sum(x2[:,np.newaxis,:,:] * z[:,:,np.newaxis,np.newaxis], 0) 
        elif self.mode == 'diagonal':
            x2 = X * X
            ss.m_var_x2 += np.sum(x2[:,np.newaxis,:] * z[:,:,np.newaxis], 0)
        elif self.mode == 'spherical':
            x2 = np.sum(X * X, 1)
            ss.m_var_x2 += np.sum(x2[:,np.newaxis] * z, 0)

    def fit(self, X, size = 200, max_iter = 1000):
        self.new_init(X)
        for i in range(max_iter):
            samples = np.array(np.random.sample(size) * X.shape[0], dtype = 'int32')
            data = X[samples]
            self.process_documents([data])

    def predict(self, X):
        Elogsticks_1st = expect_log_sticks(self.m_var_sticks) 
        res = self.E_log_gauss(X) + Elogsticks_1st
        return res.argmax(axis=1)

    def E_log_gauss(self, X):
        if self.mode == 'full':
            const = np.ones(self.m_T) * (- self.m_dim * np.log(2 * np.pi))
            const += self.m_log_det_precis
            const -= self.m_dim * (np.log(0.5 * self.m_rel) + 1 + 1 / self.m_rel)
            for d in range(self.m_dim):
                const += digamma(0.5 * (self.m_rel - d))
            const += np.sum(np.sum(self.m_precis * (self.m_cov - self.m_means[:,:,np.newaxis] * \
                self.m_means[:,np.newaxis,:]), axis = -1), -1)
            Elog = -np.sum(np.sum(self.m_precis[np.newaxis,:,:,:] * X[:,np.newaxis,:,np.newaxis] * \
                X[:,np.newaxis,np.newaxis,:], -1), -1)
            Elog += 2 * np.sum(np.sum(self.m_precis[np.newaxis,:,:,:] * X[:, np.newaxis,:, np.newaxis] * \
                self.m_means[np.newaxis, :, np.newaxis, :], -1), -1)
            Elog += const[np.newaxis, :]
            Elog *= 0.5

        elif self.mode == 'diagonal':
            t = self.m_rel * 0.5 + 1
            Elog = -0.5 * t[np.newaxis, :] * np.sum(X[:,np.newaxis,:] * X[:,np.newaxis,:] / self.m_cov[np.newaxis,:,:], 2)
            Elog += t[np.newaxis,:] * np.sum(self.m_means[np.newaxis,:,:] / self.m_cov[np.newaxis,:,:] * X[:,np.newaxis,:], 2)
            const = 0.5 * (digamma(t) * self.m_dim - \
                np.sum(np.log(self.m_cov), 1) - \
                t * np.sum(self.m_means * self.m_means / self.m_cov, 1) - \
                self.m_dim / self.m_rel - self.m_dim * np.log(2 * np.pi))
            Elog += const[np.newaxis, :]

        elif self.mode == 'spherical':
            t = 0.5 * (self.m_dim * self.m_rel - self.m_dim + 2)
            const = 0.5 * (self.m_dim * np.log(2 * np.pi) + self.m_dim * digamma(t) - self.m_dim * np.log(self.m_cov) - t * np.sum(self.m_means * self.m_means, 1) / self.m_cov - self.m_dim / self.m_rel)
            Elog = t * np.sum(self.m_means[np.newaxis,:,:] / self.m_cov[np.newaxis,:,np.newaxis] * X[:,np.newaxis,:], 2)
            Elog -= 0.5 * t / self.m_cov[np.newaxis, :] * np.sum(X * X,1)[:,np.newaxis]
        return Elog

    def update_model(self, sstats):
        # rhot will be between 0 and 1, and says how much to weight
        # the information we got from this mini-batch.
        rhot = pow(self.m_tau + self.m_updatect, -self.m_kappa)
        if rhot < rhot_bound: 
            rhot = rhot_bound
        self.m_rhot = rhot

        self.m_updatect += 1

        scale = self.m_total / sstats.m_batchsize
        self.m_varphi_ss = (1.0-rhot) * self.m_varphi_ss + rhot * \
               sstats.m_var_sticks_ss * scale

        var_x2, var_x = self.par_to_natual(self.m_cov, self.m_means, self.m_rel)
        self.m_rel = self.m_rel * (1 - rhot) + rhot * (self.m_rel0 + scale * sstats.m_var_res)

        var_x = var_x * (1 - rhot) + rhot * (self.m_var_x0 + scale * sstats.m_var_x)
        var_x2 = var_x2 * (1 - rhot) + rhot * (self.m_var_x20 + scale * sstats.m_var_x2)
        self.m_cov, self.m_means = self.natual_to_par(var_x2, var_x, self.m_rel)
        if self.mode == 'full':
            for t in range(self.m_T):
                self.m_precis[t] = linalg.inv(self.m_cov[t])
                self.m_log_det_precis[t] = linalg.det(self.m_precis[t])

        ## update top level sticks 
        var_sticks_ss = np.zeros((2, self.m_T-1))
        self.m_var_sticks[0] = self.m_varphi_ss[:self.m_T-1]  + 1.0
        var_phi_sum = np.flipud(self.m_varphi_ss[1:])
        self.m_var_sticks[1] = np.flipud(np.cumsum(var_phi_sum)) + self.m_gamma

    def save_model(self, output):
        model = {'sticks':self.m_var_sticks,
                'means': self.m_means,
                'precis':self.m_cov}
        cPickle.dump(model, output)

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

    def doc_e_step(self, X, ss, Elogsticks_1st, var_converge, max_iter=100):
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

        #TODO
        #while iter < max_iter and (converge <= 0.0 or converge > var_converge):
        while iter < max_iter:
            ### update variational parameters
            # var_phi 
            if iter < 3:
                var_phi = np.dot(phi.T, Eloggauss)
                (log_var_phi, log_norm) = log_normalize(var_phi)
                var_phi = np.exp(log_var_phi)
            else:
                var_phi = np.dot(phi.T,  Eloggauss) + Elogsticks_1st
                (log_var_phi, log_norm) = log_normalize(var_phi)
                var_phi = np.exp(log_var_phi)
            
            # phi
            if iter < 3:
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
            likelihood += np.sum((np.array([1.0, self.m_alpha])[:,np.newaxis]-v) * (sp.psi(v)-dig_sum))
            likelihood -= np.sum(sp.gammaln(np.sum(v, 0))) - np.sum(sp.gammaln(v))

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
            
        # update the suff_stat ss 
        z = np.dot(phi, var_phi) 
        self.add_to_sstats(var_phi, z, X, ss)
        return likelihood
