import numpy as np
import scipy.special as sp
import os, sys, math, time
import utils
from corpus import document, corpus
from itertools import izip
import random
import cPickle

from scipy.special import digamma as _digamma, gammaln as _gammaln

meanchangethresh = 0.00001
random_seed = 999931111
np.random.seed(random_seed)
random.seed(random_seed)
min_adding_noise_point = 10
min_adding_noise_ratio = 1 
mu0 = 0.3
rhot_bound = 0.0

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
    def __init__(self, T, Wt, Dt):
        self.m_batchsize = Dt
        self.m_var_sticks_ss = np.zeros(T) 
        self.m_var_beta_ss = np.zeros((T, Wt))
        # Wt is the vector's dimension
        self.m_var_mean_1 = np.zeros((self.m_T, self.m_dim))
        self.m_var_mean_2 = np.zeros(self.m_T)
        self.m_var_prec_1 = np.zeros(self.m_T)
        self.m_var_prec_2 = np.zeros(self.m_T)
    
    def set_zero(self):
        self.m_var_sticks_ss.fill(0.0)
        self.m_var_beta_ss.fill(0.0)
        self.m_var_mean_ss.fill(0.0)

class online_hdp:
    ''' hdp model using stick breaking'''
    def __init__(self, T, K, D, alpha, gamma, kappa, tau, dim = 500, scale=1.0, adding_noise=False):
        """
        gamma: first level concentration
        alpha: second level concentration
        T: top level truncation level
        K: second level truncation level
        D: number of documents in the corpus
        kappa: learning rate
        tau: slow down parameter
        """
        self.m_D = D # number of corps
        self.m_T = T # Top level truncation
        self.m_K = K # second level truncation
        self.m_alpha = alpha # second level concentration
        self.m_gamma = gamma # first level truncation

        ## each column is the top level topic's beta distribution para
        self.m_var_sticks = np.zeros((2, T-1))
        self.m_var_sticks[0] = 1.0
        #self.m_var_sticks[1] = self.m_gamma
        # make a uniform at beginning
        self.m_var_sticks[1] = range(T-1, 0, -1)

        self.m_varphi_ss = np.zeros(T)

        ## start
        self.m_dim = dim # the vector dimension
        self.m_means = np.random.uniform(0.000001, 10, (self.m_T - 1, self.m_dim))
        self.m_dof = np.ones(self.m_T - 1)
        self.m_scale = np.ones(self.m_T - 1)
        self.m_precs = np.ones((self.m_T, self.m_dim))
        self.bound_prec = 0.5 * self.m_T * (digamma(self.m_dof) - np.log(self.m_scale))

        self.m_tau = tau + 1
        self.m_kappa = kappa
        self.m_scale = scale
        self.m_updatect = 0 
        self.m_status_up_to_date = True
        self.m_adding_noise = adding_noise
        self.m_num_docs_parsed = 0

        # Timestamps and normalizers for lazy updates
        self.m_r = [0]

    def new_init(self, c):
        """ need implement"""
        self.means_ = cluster.KMeans(
            n_clusters=self.n_components,
            random_state=self.random_state).fit(X).cluster_centers_[::-1]

    def process_documents(self, docs, var_converge, unseen_ids=[], ropt_o=True):
        ss = suff_stats(self.m_T, Wt, len(docs)) 
        Elogsticks_1st = expect_log_sticks(self.m_var_sticks) # global sticks

        # run variational inference on some new docs
        score = 0.0
        count = 0
        unseen_score = 0.0
        unseen_count = 0
        for i, cop in enumerate(cops):
            cop_score = self.doc_e_step(cop, ss, Elogsticks_1st, var_converge)

        self.update_model(ss)
    
        return (score, count, unseen_score, unseen_count) 

    def doc_e_step(self, cop, ss, Elogsticks_1st, \
                   word_list, unique_words, var_converge, \
                   max_iter=100):
        """
        e step for a single corps
        """

        ## Elogbeta_doc = self.m_Elogbeta[:, doc.words] 
        ## very similar to the hdp equations
        v = np.zeros((2, self.m_K-1))  
        v[0] = 1.0
        v[1] = self.m_alpha

        # The following line is of no use.
        Elogsticks_2nd = expect_log_sticks(v)

        # back to the uniform
        phi = np.ones((cop.shape[0], self.m_K)) * 1.0/self.m_K

        likelihood = 0.0
        old_likelihood = -1e100
        converge = 1.0 
        eps = 1e-100
        
        iter = 0

        diff2 = np.array((X.shape[0], self.m_T))
        for s in range(X.shape[0]):
            diff2[s] = np.sum((X[s] - self.m_means) ** 2, axis = 1)
        #Eloggauss:P(X|topic k), shape: sample, topic
        Eloggauss = np.array((X.shape[0], self.m_T))
        Eloggauss -= (diff2 + self.m_dim) * 0.5 * self.m_dof / self.m_scale
        Eloggauss += self.m_dim * 0.5 * (digamma(self.m_dof) - np.log(self.m_scale))
        Eloggauss -= 0.5 * self.m_dim * np.log(2 * np.pi) + np.log(2 * np.pi * np.e)

        # not yet support second level optimization yet, to be done in the future
        while iter < max_iter and (converge < 0.0 or converge > var_converge):
            ### update variational parameters
            # var_phi 
            if iter < 3:
                var_phi = np.dot(phi.T, Eloggauss)
                (log_var_phi, log_norm) = utils.log_normalize(var_phi)
                var_phi = np.exp(log_var_phi)
            else:
                var_phi = np.dot(phi.T,  Eloggauss) + Elogsticks_1st
                (log_var_phi, log_norm) = utils.log_normalize(var_phi)
                var_phi = np.exp(log_var_phi)
            
            # phi
            if iter < 3:
                phi = np.dot(Eloggauss, var_phi.T)
                (log_phi, log_norm) = utils.log_normalize(phi)
                phi = np.exp(log_phi)
            else:
                phi = np.dot(Eloggauss, var_phi.T) + Elogsticks_2nd
                (log_phi, log_norm) = utils.log_normalize(phi)
                phi = np.exp(log_phi)

            # v
            #phi_all = phi * np.array(doc.counts)[:,np.newaxis]
            phi_all = phi[:, np.newaxis]
            v[0] = 1.0 + np.sum(phi_all[:,:self.m_K-1], 0)
            phi_cum = np.flipud(np.sum(phi_all[:,1:], 0))
            v[1] = self.m_alpha + np.flipud(np.cumsum(phi_cum))
            Elogsticks_2nd = expect_log_sticks(v)

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

            converge = (likelihood - old_likelihood)/abs(old_likelihood)
            old_likelihood = likelihood

            if converge < -0.000001:
                print "warning, likelihood is decreasing!"
            
            iter += 1
            
        # update the suff_stat ss 
        # this time it only contains information from one doc
        ss.m_var_sticks_ss += np.sum(var_phi, 0)   
        ss.m_var_beta_ss[:, batchids] += np.dot(var_phi.T, phi.T * doc.counts)
        z = np.dot(var_phi, phi)
        ss.m_var_mean_1 = np.dot(z.T, cop)
        prs = (self.m_dof * self.m_scale)[:, np.newaxis]
        ss.m_var_mean_1 *= prs
        ss.m_var_mean_2 = np.sum(z * prs, axis = 0)

        ss.m_var_prec_1 = np.sum(z, axis = 0)
        ss.m_var_prec_2 = np.sum(z * diff2, axis = 0)

        return(likelihood)

    def update_model(self, sstats):
        self.m_status_up_to_date = False
        # rhot will be between 0 and 1, and says how much to weight
        # the information we got from this mini-batch.
        rhot = self.m_scale * pow(self.m_tau + self.m_updatect, -self.m_kappa)
        if rhot < rhot_bound: 
            rhot = rhot_bound
        self.m_rhot = rhot

        self.m_updatect += 1
        self.m_r.append(self.m_r[-1] + np.log(1-rhot))

        self.m_varphi_ss = (1.0-rhot) * self.m_varphi_ss + rhot * \
               sstats.m_var_sticks_ss * self.m_D / sstats.m_batchsize

        means = sstats.m_var_mean_1 / (sstats.m_var_mean_2 + 1)[:,np.newaxis]
        self.m_means = (1 - rhot) * self.m_means \
            + rhot * self.m_D * means / sstats.m_batchsize

        dof = 1 + self.m_T * ss.m_var_prec_1
        scale = 1 + 0.5 * (ss.m_var_prec_2 + ss.m_var_prec_1 * self.m_T)

        self.m_dof = self.m_dof * (1 - rhot) + \
            rhot * dof * self.m_D / sstats.m_batchsize
        self.m_scale = self.m_scale * (1 - rhot) + \
            rhot * scale * self.m_D / sstats.m_batchsize

        ## update top level sticks 
        var_sticks_ss = np.zeros((2, self.m_T-1))
        self.m_var_sticks[0] = self.m_varphi_ss[:self.m_T-1]  + 1.0
        var_phi_sum = np.flipud(self.m_varphi_ss[1:])
        self.m_var_sticks[1] = np.flipud(np.cumsum(var_phi_sum)) + self.m_gamma

    def save_model(self, output):
        model = {'sticks':self.m_var_sticks,
                'means': self.m_means,
                'dof':self.m_dof,
                'scale':self.m_scale}
        
        cPickle.dump(model, output)
