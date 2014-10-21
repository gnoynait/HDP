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
        self.m_var_x2 = np.zeros((T, dim, dim))
        self.m_var_res = np.zeros(T)

class online_hdp:
    ''' hdp model using stick breaking'''
    def __init__(self, T, K, D, alpha, gamma, kappa, tau, total, dim = 500):
        """
        gamma: first level concentration
        alpha: second level concentration
        T: top level truncation level
        K: second level truncation level
        D: number of documents in the corpus
        kappa: learning rate
        tau: slow down parameter
        """
        self.m_T = T # Top level truncation
        self.m_K = K # second level truncation
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
        self.m_rel0 = self.m_dim + 2 
        self.m_var_x0 = np.zeros(self.m_dim)
        self.m_var_x20 = np.tile(np.eye(self.m_dim) * self.m_rel0, (self.m_T, 1, 1))
        ## for gaussian
        ## TODO random init these para
        self.m_means = np.random.normal(0, 0.2, (self.m_T, self.m_dim))
        self.m_precis = np.tile(np.eye(self.m_dim), (self.m_T, 1, 1))
        self.m_rel = np.ones(self.m_T) * self.m_rel0
        self.m_var_x2, self.m_var_x = self.par_to_natual(self.m_precis, self.m_means, self.m_rel)
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

    def natual_to_par(self, x2, x, r):
        mean = x / r[:, np.newaxis]
        x2 = x2 / r[:,np.newaxis, np.newaxis]
        precis = x2 - mean[:,:,np.newaxis] * mean[:,np.newaxis,:]
        return precis, mean

    def par_to_natual(self, precis, mean, r):
        x = mean * r[:, np.newaxis]
        x2 = precis + mean[:, :, np.newaxis] * mean[:, np.newaxis, :]
        x2 = x2 * r[:, np.newaxis, np.newaxis]
        return x2, x
        
    def new_init(self, c):
        """ need implement"""
        """
        self.m_means = cluster.KMeans(
            n_clusters=self.m_T,
            random_state=self.random_state).fit(c).cluster_centers_[::-1]
        """
        np.random.shuffle(c)
        self.m_means[:] = c[0:self.m_T]
        self.m_var_x2, self.m_var_x = self.par_to_natual(self.m_precis, self.m_means, self.m_rel)
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
        
        diff2 = self.square_diff(X)
        # TODO: diff2 is wrong
        #for s in range(X.shape[0]):
        #    diff2[s] = np.sum((X[s] - self.m_means) ** 2, axis = 1)
        #Eloggauss:P(X|topic k), shape: sample, topic
        Eloggauss = self.E_log_gauss_diff2(diff2)

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
            
        #debug(iter, converge)
        # update the suff_stat ss 
        z = np.dot(phi, var_phi) 
        ss.m_var_sticks_ss += np.sum(var_phi, 0)   

        #debug(phi)
        #debug(var_phi)
        #debug(z)
        ss.m_var_res += np.sum(z, axis = 0)
        #x2 = np.zeros((X.shape[0], self.m_dim, self.m_dim))
        x2 = X[:,:,np.newaxis] * X[:,np.newaxis,:]
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
        ss.m_var_x2 += np.sum(x2[:,np.newaxis,:,:] * z[:,:,np.newaxis,np.newaxis], axis = 0) 
        return likelihood

    def fit(self, X):
        self.new_init(X)
        size = X.shape[0] #/ 5
        for i in range(500):
            np.random.shuffle(X)
            data = X[:size]
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
        diff2 = np.sum(diff[:,:,:,np.newaxis] * self.m_precis[np.newaxis,:,:,:] * diff[:,:,np.newaxis,:], axis = (2, 3))
        return diff2

    def E_log_gauss(self, X):
        diff2 = self.square_diff(X)
        return self.E_log_gauss_diff2(diff2)

    def E_log_gauss_diff2(self, diff2):
        ## approximate
        ## _TODO implement exact computation
        ##return self.log_gauss_diff2(diff2)
        const = np.ones(self.m_T) * (-0.5 * self.m_dim * np.log(2 * np.pi))
        Elogdetp = np.ones(self.m_T) * (self.m_dim * np.log(2))
        for t in range(self.m_T):
            ## TODO: TODO TODO here is wrong
            ## should plus the det of cov, not precis!!!!!
            ## need to verify the problem
            Elogdetp[t] += 0.5 * np.log(linalg.det(self.m_precis[t]/self.m_rel[t]))
            if np.isnan(Elogdetp[t]):
                print self.m_precis[t]
                print linalg.det(self.m_precis[t]/self.m_rel[t])
                sys.exit()

        for d in range(self.m_dim):
            Elogdetp += digamma((self.m_rel -d) / 2)
        const += 0.5 * Elogdetp

        Eloggauss = np.zeros((diff2.shape[0], self.m_T))
        Eloggauss -= 0.5 * diff2
        Eloggauss -= 0.5 * np.sum(self.m_precis.reshape((self.m_T, -1)), axis=1)
        Eloggauss -= const
        return Eloggauss

    def log_gauss_diff2(self, diff2):
        const = np.ones(self.m_T) * (-0.5 * self.m_dim * np.log(2 * np.pi))
        for t in range(self.m_T):
            const[t] += 0.5 * np.log(linalg.det(self.m_precis[t]))
        Eloggauss = np.zeros((diff2.shape[0], self.m_T))
        Eloggauss -= 0.5 * diff2
        Eloggauss -= const
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
        self.m_rel = self.m_rel * (1 - rhot) + rhot * (self.m_rel0 + scale * sstats.m_var_res)
        self.m_var_x = self.m_var_x * (1 - rhot) + rhot * (self.m_var_x0 + scale * sstats.m_var_x)
        self.m_var_x2 = self.m_var_x2 * (1 - rhot) + rhot * (self.m_var_x20 + scale * sstats.m_var_x2.reshape((-1, self.m_dim, self.m_dim)))
        """
        self.m_means = self.m_var_x / self.m_rel[:, np.newaxis]
        for t in range(self.m_T):
            self.m_precis[t] = self.m_var_x2[t] / self.m_rel[t] - \
                np.dot(self.m_means[t][:, np.newaxis], self.m_means[t][np.newaxis,:])
        """
        self.m_precis, self.m_means = self.natual_to_par(self.m_var_x2, self.m_var_x, self.m_rel)

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
