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

from onlinedpgmm import *

random_seed = int(time.time())
np.random.seed(random_seed)
random.seed(random_seed)

class online_hdp(online_dp):
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
        online_dp.__init__(self, T, gamma, kappa, tau, total, dim)
        self.m_K = K # second level truncation
        self.m_alpha = alpha # second level concentration
        print 'deperated'
        sys.exit()

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
        ss.m_var_sticks_ss += np.sum(var_phi, 0)   

        ss.m_var_res += np.sum(z, axis = 0)
        x2 = X[:,:,np.newaxis] * X[:,np.newaxis,:]
        ss.m_var_x += np.sum(X[:,np.newaxis,:] * z[:,:,np.newaxis], axis = 0)
        ss.m_var_x2 += np.sum(x2[:,np.newaxis,:,:] * z[:,:,np.newaxis,np.newaxis], axis = 0) 
        return likelihood

    def save_model(self, output):
        model = {'sticks':self.m_var_sticks,
                'means': self.m_means,
                'precis':self.m_precis}
        cPickle.dump(model, output)
