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

class FullFactorSpheGaussianMixture:
    def __init__(self, k, dim, mean_prior=(1, 0), prec_prior = (2, 2)):
        self.dim = dim
        self.mean_prior0, self.mean_prior1 = mean_prior
        self.mean_post0 = np.ones(k)
        self.mean_post1 = np.random.randn(k, dim)
        self.prec_prior0, self.prec_prior2 = prec_prior
        self.prec_post0 = np.ones(k) * prec_prior[0]
        self.prec_post2 = np.ones(k) * prec_prior[1]
        self._updateExpectation()

    def update(self, data, assign, rate):
        stat0 = np.sum(assign, axis=0)
        stat1 = np.dot(assign.T, data)
        stat2 = np.sum(
            np.square(data[:,np.newaxis,:]-self.expc_mean[np.newaxis,:,:]), 
            axis=(0, 2))
        self.mean_post1 = rate*(self.mean_prior1 + stat1) + (1.0-rate)*self.mean_post1
        self.mean_post0 = rate*(self.mean_prior0 + stat0) + (1.0-rate)*self.mean_post0
        self.prec_post2 = rate*(self.prec_prior2 + stat2) + (1.0-rate)*self.prec_post2
        self.prec_post0 = rate*(self.prec_prior0 + stat2) + (1.0-rate)*self.prec_post0
        self._updateExpectation()

    def _updateExpectation(self):
        self.expc_mean = self.mean_post1 / self.mean_post0[:,np.newaxis]
        self.expc_prec = (self.prec_post2 + 2.0) / self.prec_post0
        self.expc_lnprec = sp.psi(self.prec_post2 / 2+ 1.0) - np.log(self.prec_post0 / 2)
        self.expc_const = 0.5 * self.expc_lnprec - self.dim * np.log(2*np.pi) - 0.5 / self.expc_prec
    def calcLogProb(self, data):
        dist = np.sum(
            np.square(data[:,np.newaxis,:]-self.expc_mean[np.newaxis,:,:]),
            axis=2)
        logprob = dist * self.expc_prec[np.newaxis, :] * 0.5 + self.expc_const[np.newaxis,:]
        return logprob

class NonBayesianWeight:
    def __init__(self, T):
        self.T = T
        self.weight = np.ones(T) / T

    def logWeight(self):
        return np.log(self.weight)

    def update(self, z, lr):
        self.weight = lr * z + (1-lr) * self.weight

class DirichletWeight:
    def __init__(self, T, alpha):
        self.T = T
        self.alpha = alpha
        self.weight = np.ones(T) * alpha

    def logWeight(self):
        return np.log(self.weight / np.sum(self.weight))

    def update(self, z, lr):
        self.weight = lr * (np.log(z) + self.alpha) + (1.0-lr) * self.weight

    
class StickBreakingWeight:
    def __init__(self, T, alpha):
        self.T = T
        self.alpha = alpha
        ## each column is the top level topic's beta distribution para
        sticks = np.zeros((2, T-1))
        sticks[0] = 1.0
        sticks[1] = alpha

        self.sticks = sticks
        self.varphi = np.zeros(T)

    def logWeight(self):
        """For stick-breaking hdp, this returns the E[log(sticks)] 
        """
        sticks = self.sticks
        dig_sum = sp.psi(np.sum(sticks, 0))
        ElogW = sp.psi(sticks[0]) - dig_sum
        Elog1_W = sp.psi(sticks[1]) - dig_sum

        n = len(sticks[0]) + 1
        Elogsticks = np.zeros(n)
        Elogsticks[0:n-1] = ElogW
        Elogsticks[1:] = Elogsticks[1:] + np.cumsum(Elog1_W)
        return Elogsticks 

    def update(self, z, lr):
        self.varphi = (1.0-lr) * self.varphi + lr * z
        ## update top level sticks 
        self.sticks[0] = self.varphi[:self.T-1]  + 1.0
        varphi_sum = np.flipud(self.varphi[1:])
        self.sticks[1] = np.flipud(np.cumsum(varphi_sum)) + self.alpha

class OnlineDP:
    """Online DP model"""
    def __init__(self, T, gamma, kappa, tau, total, dim, model):
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
        self.sticks = StickBreaking(alpha, T)



        self.m_dim = dim # the vector dimension
        ## mode: spherical, diagonal, full
        self.model = model

    def assign(self, X):
        likelihood = 0.0
        old_likelihood = -1e100
        converge = 1.0 
        eps = 1e-100
        iter = 0

        
        Eloggauss = self.model.calcLogProb(X)
        z = Eloggauss + self.sticks.expectLogSticks()
        z, norm = log_normalize(z)
        z = np.exp(z)
        # varphi equals to z

        return z, likelihood

    def predict(self, X):
        logLik = self.model.calcLogProb(X)
        post = logLik + self.sticks.expectLogSticks()
        return post.argmax(axis=1)

    def update(self, X, z, lr):
        # rhot will be between 0 and 1, and says how much to weight
        # the information we got from this mini-batch.

        scale = self.m_total / sstats.batchsize

        self.sticks.update(z, rate)
        self.model.update(X, z, rate)

    def save_model(self, output):
        model = {'sticks':self.var_stick,
                'means': self.m_mean,
                'precis':self.m_cov}
        cPickle.dump(model, output)

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
    def __init__(self, model, lrShedule):
        self.sheduler = lrShedule
        self.model = model
    def fit(self, DataSource):
        logs = []
        for batch in DataSource:
            assign, loglik = self.model.assign(batch)
            logs.append(loglik)
            self.model.update(batch, assign, self.sheduler.nextRate())
        return np.array(logs)

class OnlineHDP:
    """Online HDP Model"""
    def __init__(self, alpha, K, T, size, batchsize, data, \
            coldstart=False, maxiter=100, online=True):
        self.m_alpha = alpha
        self.m_K = K # second level
        self.m_T = T # first level
        v = np.zeros((2, self.m_K - 1))
        v[0] = 1.0
        v[1] = alpha
        self.v = v
        self.varphi = np.zeros((K, T)) # K * T array
        self.size = size # don't need to be the same the data
        self.batchsize = batchsize

    def update(self, data, lr):
        Elogsticks_1st = expect_log_sticks(self.model.var_sticks)
        X = group.sample()

        Elogsticks_2nd = expect_log_sticks(self.v)
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

    def assign(self, X):
        z, _ = self.model.assign(X)
        eLogSticks = expect_log_sticks(self.v)
        z = np.dot(phi, varphi) 

    def update(self, X, rate):
        self.model.update(X, rate)
        # update sticks
        pass
        

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

