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

#meanchangethresh = 0.00001
random_seed = 999931111
#random_seed = int(time.time())
np.random.seed(random_seed)
random.seed(random_seed)
#min_adding_noise_point = 10
#min_adding_noise_ratio = 1 
#mu0 = 0.3
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
    def __init__(self, T, dim, size):
        # T: top level topic number
        # dim: dimension
        # size: batch size
        self.m_batchsize = size
        self.m_var_sticks_ss = np.zeros(T) 
        self.m_var_x = np.zeros((T, dim))
        self.m_var_x2 = np.zeros((T, dim, dim))
        self.m_var_res = np.zeros(T)

class online_dp:
    ''' hdp model using stick breaking'''
    def __init__(self, T, K, D, alpha, gamma, kappa, tau, total, dim = 500):
        """
        gamma: first level concentration
        alpha: second level concentration
        T: top level truncation level
        D: number of documents in the corpus
        kappa: learning rate
        tau: slow down parameter
        """
        self.m_T = T # Top level truncation
        self.m_alpha = alpha # second level concentration
        self.m_gamma = gamma # first level truncation
        self.m_total = total # total ponits

        ## each column is the top level topic's beta distribution para
        self.m_var_sticks = np.zeros((2, T-1))
        self.m_var_sticks[0] = 1.0
        self.m_var_sticks[1] = self.m_gamma

        self.m_varphi_ss = np.zeros(T)

        self.m_dim = dim # the vector dimension
        ## the prior of each gaussian
        self.m_rel0 = np.ones(self.m_T) * (self.m_dim + 2)
        self.m_var_x0 = np.zeros((self.m_T, self.m_dim))
        self.m_var_x20 = np.tile(np.eye(self.m_dim), (self.m_T, 1, 1)) \
            * self.m_rel0[:, np.newaxis, np.newaxis]
        ## for gaussian
        self.m_means = np.random.normal(0, 1, (self.m_T, self.m_dim))
        self.m_precis = np.tile(np.eye(self.m_dim), (self.m_T, 1, 1))
        self.m_rel = np.ones(self.m_T) * self.m_rel0

        self.m_tau = tau + 1
        self.m_kappa = kappa
        self.m_updatect = 0 

    def natual_to_par(self, x2, x, r, msg='unknow'):
        mean = x / r[:, np.newaxis]
        cov = x2 / r[:,np.newaxis, np.newaxis] - mean[:,:,np.newaxis] * mean[:,np.newaxis,:]
        precis = np.empty(cov.shape)
        for t in range(self.m_T):
            if linalg.det(cov[t]) < 0:
                print msg
                #print cov[t]
                s =  x2[t] / r[t]
                a =  x[t]
                m = x[t] / r[t]
                #print s
                #print m
                c =  s - a[:, np.newaxis] * a[np.newaxis,:] / (r[t] * r[t])
                #print c
                #print s - m[:,np.newaxis] * m[np.newaxis,:]
                print linalg.det(c)
                #print r[t]
                #sys.exit()
            precis[t] = linalg.inv(cov[t])
        return precis, mean

    def par_to_natual(self, precis, mean, r):
        x = mean * r[:, np.newaxis]
        cov = np.empty(precis.shape)
        for t in range(self.m_T):
            cov[t] = linalg.inv(precis[t])
        x2 = cov + mean[:, :, np.newaxis] * mean[:, np.newaxis, :]
        x2 = x2 * r[:, np.newaxis, np.newaxis]
        return x2, x
        
    def new_init(self, c):
        np.random.shuffle(c)
        self.m_means[:] = c[0:self.m_T]

    def process_documents(self, cops, var_converge = 0.000001):
        size = 0
        for c in cops:
            size += c.shape[0]
        ss = suff_stats(self.m_T, self.m_dim, size) 
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
        ss.m_var_sticks_ss += np.sum(z, 0)   

        ss.m_var_res += np.sum(z, axis = 0)
        x2 = X[:,:,np.newaxis] * X[:,np.newaxis,:]
        ss.m_var_x += np.sum(X[:,np.newaxis,:] * z[:,:,np.newaxis], axis = 0)
        ss.m_var_x2 += np.sum(x2[:,np.newaxis,:,:] * z[:,:,np.newaxis,np.newaxis], axis = 0) 
        return likelihood

    def fit(self, X):
        self.new_init(X)
        size = 500
        for i in range(500):
            samples = np.array(np.random.sample(size) * X.shape[0], dtype = 'int32')
            data = X[samples]
            self.process_documents([data])

    def predict(self, X):
        res = self.E_log_gauss(X)
        return res.argmax(axis=1)

    def E_log_gauss(self, X):
        """ full """
        cov = np.empty(self.m_precis.shape)
        const = np.ones(self.m_T) * (- self.m_dim * np.log(2 * np.pi))
        for t in range(self.m_T):
            cov[t] = linalg.inv(self.m_precis[t])
            const[t] += np.log(linalg.det(self.m_precis[t])) 
            if np.isnan(const[t]):
                """
                print self.m_precis[t]
                print linalg.inv(self.m_precis[t])
                print linalg.det(self.m_precis[t])
                print self.m_rel[t]
                """
                sys.exit()

        const -= self.m_dim * (np.log(0.5 * self.m_rel) + 1 + 1 / self.m_rel)
        for d in range(self.m_dim):
            const += digamma(0.5 * (self.m_rel - d))
        const += np.sum(self.m_precis * (cov - self.m_means[:,:,np.newaxis] * \
            self.m_means[:,np.newaxis,:]), axis = (1, 2))
        Elog = -np.sum(self.m_precis[np.newaxis,:,:,:] * X[:,np.newaxis,:,np.newaxis] * \
            X[:,np.newaxis,np.newaxis,:], (2, 3))
        Elog += 2 * np.sum(self.m_precis[np.newaxis,:,:,:] * X[:, np.newaxis,:, np.newaxis] * \
            self.m_means[np.newaxis, :, np.newaxis, :], (2, 3))
        Elog += const[np.newaxis, :]
        return 0.5 * Elog

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

        self.m_rel = self.m_rel * (1 - rhot) + rhot * (self.m_rel0 + scale * sstats.m_var_res)
        var_x2, var_x = self.par_to_natual(self.m_precis, self.m_means, self.m_rel)
        self.natual_to_par(var_x2, var_x, self.m_rel, 'init')
        #print sstats.m_var_x
        #print sstats.m_var_x2
        var_x = var_x * (1 - rhot) + rhot * (self.m_var_x0 + scale * sstats.m_var_x)
        var_x2 = var_x2 * (1 - rhot) + rhot * (self.m_var_x20 + scale * sstats.m_var_x2)
        self.natual_to_par(sstats.m_var_x2, sstats.m_var_x, sstats.m_var_res, 'test')
        self.natual_to_par(self.m_var_x20, self.m_var_x0, self.m_rel0, 'rel0')
        self.m_precis, self.m_means = self.natual_to_par(var_x2, var_x, self.m_rel, 'update')

        ## update top level sticks 
        var_sticks_ss = np.zeros((2, self.m_T-1))
        self.m_var_sticks[0] = self.m_varphi_ss[:self.m_T-1]  + 1.0
        var_phi_sum = np.flipud(self.m_varphi_ss[1:])
        self.m_var_sticks[1] = np.flipud(np.cumsum(var_phi_sum)) + self.m_gamma
        #debug(np.exp(expect_log_sticks(self.m_var_sticks)))

    def save_model(self, output):
        model = {'sticks':self.m_var_sticks,
                'means': self.m_means,
                'precis':self.m_precis}
        cPickle.dump(model, output)
