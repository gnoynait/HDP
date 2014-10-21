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

#meanchangethresh = 0.00001
#random_seed = 999931111
random_seed = int(time.time())
np.random.seed(random_seed)
random.seed(random_seed)
#min_adding_noise_point = 10
#min_adding_noise_ratio = 1 
#mu0 = 0.3
rhot_bound = 0.0

def debug(*W):
    for w in W:
        print w
    print '-' * 75

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
        self.m_var_res = np.zeros(T)
        self.m_var_x2 = np.zeros(T)

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
        # make a uniform at beginning
        #self.m_var_sticks[1] = range(T-1, 0, -1)

        self.m_varphi_ss = np.zeros(T)

        self.m_dim = dim # the vector dimension
        ## the prior of each gaussian
        #self.m_means0 = np.zeros(self.m_dim)
        #self.m_precis0 = np.eye(self.m_dim)
        self.m_rel0 = 0.1
        self.m_var_x0 = np.zeros(self.m_dim)
        ## for gaussian
        ## TODO random init these para
        # for means
        self.m_means = np.random.normal(0, 0.2, (self.m_T, self.m_dim))
        self.m_precis = np.ones(self.m_T) * self.m_rel0
        self.m_var_x = self.m_means * self.m_precis[:, np.newaxis]
        # for precis
        self.m_pre_a = 2 * np.ones(self.m_T)
        self.m_pre_b = 2 * np.ones(self.m_T)
        self.m_var_x2 = -self.m_pre_b
        self.m_var_x20 = -self.m_pre_b
        """
        self.m_var_x = self.m_means * self.m_rel[:, np.newaxis]
        self.m_var_x2 = self.m_precis.copy()
        for t in range(self.m_T):
        	self.m_var_x2[t] += np.dot(self.m_means[t][:,np.newaxis], self.m_means[t][np.newaxis,:])
        self.m_var_x2 *= self.m_rel0
        """

        self.m_tau = tau + 1
        self.m_kappa = kappa
        self.m_updatect = 0 

    def new_init(self, c):
        """ need implement"""
        """
        self.m_means = cluster.KMeans(
            n_clusters=self.m_T,
            random_state=self.random_state).fit(c).cluster_centers_[::-1]
        """
        np.random.shuffle(c)
        self.m_means[:] = c[0:self.m_T]
        self.m_var_x = self.m_means * self.m_precis[:, np.newaxis]
        """
        self.m_var_x2 = self.m_precis.copy()
        for t in range(self.m_T):
        	self.m_var_x2[t] += np.dot(self.m_means[t][:,np.newaxis], self.m_means[t][np.newaxis,:])
        self.m_var_x2 *= self.m_rel0
        """

    def process_documents(self, cops, var_converge = 0.000001):
        size = 0
        for c in cops:
            size += c.shape[0]
        ss = suff_stats(self.m_T, self.m_dim, size) 
        Elogsticks_1st = expect_log_sticks(self.m_var_sticks) # global sticks

        # run variational inference on some new docs
        score = 0.0
        for i, cop in enumerate(cops):
            cop_score = self.doc_e_step(cop, ss, Elogsticks_1st, var_converge)
            score += cop_score

        self.update_model(ss)
    
        return score

    def doc_e_step(self, X, ss, Elogsticks_1st, var_converge, max_iter=100):
        """
        e step for a single corps
        """

        likelihood = 0.0
        old_likelihood = -1e100
        converge = 1.0 
        eps = 1e-100
        iter = 0
        
        diff2 = self.square_diff(X)
        # TODO: diff2 is wrong
        #for s in range(X.shape[0]):
        #    diff2[s] = np.sum((X[s] - self.m_means) ** 2, axis = 1)
        #Eloggauss:P(X|topic k), shape: sample, topic
        Eloggauss = self.E_log_gauss_diff2(diff2)
        z = Eloggauss + Elogsticks_1st
        z, norm = log_normalize(z)
        z = np.exp(z)
        #ss.m_var_sticks_ss += np.sum(var_phi, 0)   
        ss.m_var_sticks_ss += np.sum(z, 0)   

        #debug(phi)
        #debug(var_phi)
        #debug(z)
        ss.m_var_res += np.sum(z, axis = 0)
        #x2 = np.zeros((X.shape[0], self.m_dim, self.m_dim))
        """
        for n in range(X.shape[0]):
            x2[n,:,:] = np.dot(X[n][:, np.newaxis], X[n][np.newaxis,:])
        for k in range(self.m_T):
            ss.m_var_x[k] += np.sum(X * z[:, k][:,np.newaxis], axis = 0) 
            t = x2.reshape((x2.shape[0], -1))
            ss.m_var_x2[k] += np.sum(t * z[:,k][:,np.newaxis], axis = 0)
        """
        # bug fix: change '=' to '+='
        ss.m_var_x += np.sum(X[:,np.newaxis,:] * z[:,:,np.newaxis], axis = 0)
        ss.m_var_x2 += np.sum(np.sum(X * X, axis = 1)[:, np.newaxis] * z[:,:], axis = 0)
        return likelihood

    def fit(self, X):
        self.new_init(X)
        size = 100
        for i in range(500):
            #np.random.shuffle(X)
            samples = np.array(np.random.sample(size) * X.shape[0], dtype = 'int32')
            data = X[samples]
            self.process_documents([data])

    def predict(self, X):
        res = self.E_log_gauss(X)
        return res.argmax(axis=1)

    def square_diff(self, X):
        ## return each the square of the distance bettween x and every mean
        """
        diff2 = np.zeros((X.shape[0], self.m_T))
        for t in range(self.m_T):
            diff = X - self.m_means[t]
            diff2[:,t] = np.sum(np.dot(diff,self.m_precis[t]) * diff, axis = 1)
        return diff2
        """

        diff = X[:,np.newaxis,:] - self.m_means[np.newaxis,:,:]
        diff2 = np.sum(diff[:,:,:,np.newaxis] * diff[:,:,np.newaxis,:], axis = (2, 3))
        return diff2 * 0.1

    def E_log_gauss(self, X):
        diff2 = self.square_diff(X)
        return self.E_log_gauss_diff2(diff2)

    def E_log_gauss_diff2(self, diff2):
        Epre = self.m_pre_a / self.m_pre_b
        print self.m_pre_b
        Elogpre = digamma(self.m_pre_a) - np.log(self.m_pre_b)
        const = -0.5 * self.m_dim / self.m_precis  * Epre -0.5 * self.m_dim * np.log(2 * np.pi) + 0.5 * self.m_dim * Elogpre
        Eloggauss = -0.5 * diff2 * Epre[np.newaxis, :] + const[np.newaxis, :]
        return Eloggauss

    def update_model(self, sstats):
        #debug(self.m_updatect)
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

        #debug(rhot)
        self.m_precis = self.m_precis * (1 - rhot) + rhot * (self.m_rel0 + scale * sstats.m_var_res)
        self.m_var_x = self.m_var_x * (1 - rhot) + rhot * (self.m_var_x0 + scale * sstats.m_var_x)
        self.m_means = self.m_var_x / self.m_precis[:, np.newaxis]

        # update precis
        self.m_pre_a = self.m_precis + 1
        self.m_var_x2 = self.m_var_x2 * (1 - rhot) * rhot * (self.m_var_x20 + sstats.m_var_x2)
        self.m_pre_b = - self.m_var_x2
        ## update top level sticks 
        var_sticks_ss = np.zeros((2, self.m_T-1))
        self.m_var_sticks[0] = self.m_varphi_ss[:self.m_T-1]  + 1.0
        var_phi_sum = np.flipud(self.m_varphi_ss[1:])
        self.m_var_sticks[1] = np.flipud(np.cumsum(var_phi_sum)) + self.m_gamma
        debug(np.exp(expect_log_sticks(self.m_var_sticks)))

    def save_model(self, output):
        model = {'sticks':self.m_var_sticks,
                'means': self.m_means,
                'precis':self.m_precis}
        cPickle.dump(model, output)
