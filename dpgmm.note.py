"""Bayesian Gaussian Mixture Models and
Dirichlet Process Gaussian Mixture Models"""
from __future__ import print_function

# Author: Alexandre Passos (alexandre.tp@gmail.com)
#         Bertrand Thirion <bertrand.thirion@inria.fr>
#
# Based on mixture.py by:
#         Ron Weiss <ronweiss@gmail.com>
#         Fabian Pedregosa <fabian.pedregosa@inria.fr>
#

import numpy as np
from scipy.special import digamma as _digamma, gammaln as _gammaln
from scipy import linalg
from scipy.spatial.distance import cdist

from ..externals.six.moves import xrange
from ..utils import check_random_state, deprecated
from ..utils.extmath import norm, logsumexp, pinvh
from .. import cluster
from .gmm import GMM


def sqnorm(v):
    return norm(v) ** 2


def digamma(x):
    return _digamma(x + np.finfo(np.float32).eps)


def gammaln(x):
    return _gammaln(x + np.finfo(np.float32).eps)


def log_normalize(v, axis=0):
    """Normalized probabilities from unnormalized log-probabilites"""
    v = np.rollaxis(v, axis)
    v = v.copy()
    v -= v.max(axis=0)
    out = logsumexp(v)
    v = np.exp(v - out)
    v += np.finfo(np.float32).eps
    v /= np.sum(v, axis=0)
    return np.swapaxes(v, 0, axis)


##############################################################################
# Variational bound on the log likelihood of each class
##############################################################################

def _bound_state_log_lik(X, initial_bound, precs, means, covariance_type):
    """Update the bound with likelihood terms, for standard covariance types"""
    n_components, n_features = means.shape
    n_samples = X.shape[0]
    bound = np.empty((n_samples, n_components))
    bound[:] = initial_bound
    for k in range(n_components):
        d = X - means[k]
        bound[:, k] -= 0.5 * np.sum(d * d * precs[k], axis=1)
    return bound


class DPGMM(GMM):
    """Variational Inference for the Infinite Gaussian Mixture Model.
    """
    def __init__(self, n_components=1, covariance_type='diag', alpha=1.0,
                 random_state=None, thresh=1e-2, verbose=False,
                 min_covar=None, n_iter=10, params='wmc', init_params='wmc'):
        self.alpha = alpha
        self.verbose = verbose
        super(DPGMM, self).__init__(n_components, covariance_type,
                                    random_state=random_state,
                                    thresh=thresh, min_covar=min_covar,
                                    n_iter=n_iter, params=params,
                                    init_params=init_params)

    def _get_precisions(self):
        """Return precisions as a full matrix."""
        return [np.diag(cov) for cov in self.precs_]

    def _get_covars(self):
        return [pinvh(c) for c in self._get_precisions()]

    def score_samples(self, X):
        """Return the likelihood of the data under the model.

        Compute the bound on log probability of X under the model
        and return the posterior distribution (responsibilities) of
        each mixture component for each element of X.

        This is done by computing the parameters for the mean-field of
        z for each observation.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.

        Returns
        -------
        logprob : array_like, shape (n_samples,)
            Log probabilities of each data point in X
        responsibilities: array_like, shape (n_samples, n_components)
            Posterior probabilities of each mixture component for each
            observation
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X[:, np.newaxis]
        z = np.zeros((X.shape[0], self.n_components))
        sd = digamma(self.gamma_.T[1] + self.gamma_.T[2])
        dgamma1 = digamma(self.gamma_.T[1]) - sd
        dgamma2 = np.zeros(self.n_components)
        dgamma2[0] = digamma(self.gamma_[0, 2]) - digamma(self.gamma_[0, 1] +
                                                          self.gamma_[0, 2])
        for j in range(1, self.n_components):
            dgamma2[j] = dgamma2[j - 1] + digamma(self.gamma_[j - 1, 2])
            dgamma2[j] -= sd[j - 1]
        dgamma = dgamma1 + dgamma2
        # Free memory and developers cognitive load:
        del dgamma1, dgamma2, sd

        p = _bound_state_log_lik(X, self._initial_bound + self.bound_prec_,
                                 self.precs_, self.means_,
                                 self.covariance_type)
        z = p + dgamma
        z = log_normalize(z, axis=-1)
        bound = np.sum(z * p, axis=-1)
        return bound, z

    def _update_concentration(self, z):
        """Update the concentration parameters for each cluster"""
        sz = np.sum(z, axis=0)
        self.gamma_.T[1] = 1. + sz
        self.gamma_.T[2].fill(0)
        for i in range(self.n_components - 2, -1, -1):
            self.gamma_[i, 2] = self.gamma_[i + 1, 2] + sz[i]
        self.gamma_.T[2] += self.alpha

    def _update_means(self, X, z):
        """Update the variational distributions for the means"""
        n_features = X.shape[1]
        for k in range(self.n_components):
            num = np.sum(z.T[k].reshape((-1, 1)) * X, axis=0)
            num *= self.precs_[k]
            den = 1. + self.precs_[k] * np.sum(z.T[k])
            self.means_[k] = num / den

    def _update_precisions(self, X, z):
        """Update the variational distributions for the precisions"""
        n_features = X.shape[1]
        self.dof_ = 0.5 * n_features * np.sum(z, axis=0)
        for k in range(self.n_components):
            # could be more memory efficient ?
            sq_diff = np.sum((X - self.means_[k]) ** 2, axis=1)
            self.scale_[k] = 1.
            self.scale_[k] += 0.5 * np.sum(z.T[k] * (sq_diff + n_features))
            self.bound_prec_[k] = (
                0.5 * n_features * (
                    digamma(self.dof_[k]) - np.log(self.scale_[k])))
        self.precs_ = np.tile(self.dof_ / self.scale_, [n_features, 1]).T

    def _do_mstep(self, X, z, params):
        """Maximize the variational lower bound

        Update each of the parameters to maximize the lower bound."""
        self._update_concentration(z)
        self._update_means(X, z)
        self._update_precisions(X, z)

    def _initialize_gamma(self):
        "Initializes the concentration parameters"
        self.gamma_ = self.alpha * np.ones((self.n_components, 3))

    def _bound_concentration(self):
        """The variational lower bound for the concentration parameter."""
        logprior = gammaln(self.alpha) * self.n_components
        logprior += np.sum((self.alpha - 1) * (
            digamma(self.gamma_.T[2]) - digamma(self.gamma_.T[1] +
                                                self.gamma_.T[2])))
        logprior += np.sum(- gammaln(self.gamma_.T[1] + self.gamma_.T[2]))
        logprior += np.sum(gammaln(self.gamma_.T[1]) +
                           gammaln(self.gamma_.T[2]))
        logprior -= np.sum((self.gamma_.T[1] - 1) * (
            digamma(self.gamma_.T[1]) - digamma(self.gamma_.T[1] +
                                                self.gamma_.T[2])))
        logprior -= np.sum((self.gamma_.T[2] - 1) * (
            digamma(self.gamma_.T[2]) - digamma(self.gamma_.T[1] +
                                                self.gamma_.T[2])))
        return logprior

    def _bound_means(self):
        "The variational lower bound for the mean parameters"
        logprior = 0.
        logprior -= 0.5 * sqnorm(self.means_)
        logprior -= 0.5 * self.means_.shape[1] * self.n_components
        return logprior

    def _bound_precisions(self):
        """Returns the bound term related to precisions"""
        logprior = 0.
        logprior += np.sum(gammaln(self.dof_))
        logprior -= np.sum(
            (self.dof_ - 1) * digamma(np.maximum(0.5, self.dof_)))
        logprior += np.sum(- np.log(self.scale_) + self.dof_
                           - self.precs_[:, 0])
        return logprior

    def _bound_proportions(self, z):
        """Returns the bound term related to proportions"""
        dg12 = digamma(self.gamma_.T[1] + self.gamma_.T[2])
        dg1 = digamma(self.gamma_.T[1]) - dg12
        dg2 = digamma(self.gamma_.T[2]) - dg12

        cz = np.cumsum(z[:, ::-1], axis=-1)[:, -2::-1]
        logprior = np.sum(cz * dg2[:-1]) + np.sum(z * dg1)
        del cz  # Save memory
        z_non_zeros = z[z > np.finfo(np.float32).eps]
        logprior -= np.sum(z_non_zeros * np.log(z_non_zeros))
        return logprior

    def _logprior(self, z):
        logprior = self._bound_concentration()
        logprior += self._bound_means()
        logprior += self._bound_precisions()
        logprior += self._bound_proportions(z)
        return logprior

    def lower_bound(self, X, z):
        """returns a lower bound on model evidence based on X and membership"""
        if self.covariance_type not in ['full', 'tied', 'diag', 'spherical']:
            raise NotImplementedError("This ctype is not implemented: %s"
                                      % self.covariance_type)

        X = np.asarray(X)
        if X.ndim == 1:
            X = X[:, np.newaxis]
        c = np.sum(z * _bound_state_log_lik(X, self._initial_bound +
                                            self.bound_prec_, self.precs_,
                                            self.means_, self.covariance_type))

        return c + self._logprior(z)

    def _set_weights(self):
        for i in xrange(self.n_components):
            self.weights_[i] = self.gamma_[i, 1] / (self.gamma_[i, 1]
                                                    + self.gamma_[i, 2])
        self.weights_ /= np.sum(self.weights_)

    def fit(self, X):
        """Estimate model parameters with the variational
        algorithm.

        For a full derivation and description of the algorithm see
        doc/dp-derivation/dp-derivation.tex

        A initialization step is performed before entering the em
        algorithm. If you want to avoid this step, set the keyword
        argument init_params to the empty string '' when when creating
        the object. Likewise, if you would like just to do an
        initialization, set n_iter=0.

        Parameters
        ----------
        X : array_like, shape (n, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.
        """
        self.random_state = check_random_state(self.random_state)

        ## initialization step
        X = np.asarray(X)
        if X.ndim == 1:
            X = X[:, np.newaxis]

        n_features = X.shape[1]
        z = np.ones((X.shape[0], self.n_components))
        z /= self.n_components

        self._initial_bound = - 0.5 * n_features * np.log(2 * np.pi)
        self._initial_bound -= np.log(2 * np.pi * np.e)

        if (self.init_params != '') or not hasattr(self, 'gamma_'):
            self._initialize_gamma()

        if 'm' in self.init_params or not hasattr(self, 'means_'):
            self.means_ = cluster.KMeans(
                n_clusters=self.n_components,
                random_state=self.random_state).fit(X).cluster_centers_[::-1]

        if 'w' in self.init_params or not hasattr(self, 'weights_'):
            self.weights_ = np.tile(1.0 / self.n_components, self.n_components)

        if 'c' in self.init_params or not hasattr(self, 'precs_'):
            self.dof_ = np.ones(self.n_components)
            self.scale_ = np.ones(self.n_components)
            self.precs_ = np.ones((self.n_components, n_features))
            self.bound_prec_ = 0.5 * n_features * (
                digamma(self.dof_) - np.log(self.scale_))
        logprob = []
        # reset self.converged_ to False
        self.converged_ = False
        for i in range(self.n_iter):
            # Expectation step
            curr_logprob, z = self.score_samples(X)
            logprob.append(curr_logprob.sum() + self._logprior(z))

            # Check for convergence.
            if i > 0 and abs(logprob[-1] - logprob[-2]) < self.thresh:
                self.converged_ = True
                break

            # Maximization step
            self._do_mstep(X, z, self.params)

        self._set_weights()

        return self

