import numpy as np

def log_normalize(v):
    """ return log(sum(exp(v)))"""
    log_max = 100.0
    max_val = np.max(v, 1)
    log_shift = log_max - np.log(v.shape[1]+1.0) - max_val
    tot = np.sum(np.exp(v + log_shift[:, np.newaxis]), 1)

    log_norm = np.log(tot) - log_shift
    v -= log_norm[:, np.newaxis]
    return (v, log_norm)


class GassianExpert(object):
    def __init__(self, n_components, dim, prior_var, prior_strength):
        self.n_components = n_components
        self.dim = dim
        self.prior_var, self.prior_strength = prior_var, prior_strength
        self.var_a, self.var_b = prior_strength, prior_var*prior_strength
        self.means_ = np.zeros((n_components, dim))
        self.vars = np.ones(n_components) * prior_var
        self.weight = np.ones(n_components) / n_components

    def fit(self, X, group_label, max_iter):
        x_min, x_max, y_min, y_max = X[:, 0].min(), X[:, 0].max(), X[:, 1].min(), X[:, 1].max()
        self.means_[:,0] = x_min + (x_max - x_min) * np.random.rand(self.n_components)
        self.means_[:,1] = y_min + (y_max - y_min) * np.random.rand(self.n_components)
        groups, group_count = np.unique(group_label, return_counts=True)
        group_weight = np.array(group_count, dtype=np.float) / X.shape[0]
        local_weight = np.ones((len(groups), self.n_components)) / self.n_components
        z = np.ones((X.shape[0], self.n_components))
        for n_iter in range(max_iter):
            print 'start iteration {}'.format(n_iter)
            for g in groups:
                gix = group_label == g
                log_lik = self.base_likely(X[gix])
                local_logz, _ = log_normalize(log_lik + np.log(local_weight[g]))
                local_z = np.exp(local_logz)
                local_weight[g, :] = local_z.sum(axis=0) / local_z.sum()
                z[gix,:] = local_z
            self.do_m_step(X, z)
        self.weight = local_weight.T.dot(group_weight)
        return local_weight

    def base_likely(self, X):
        log_lik = np.ones((X.shape[0], self.n_components)) / self.n_components
        for c in range(self.n_components):
            distance = np.square((X - self.means_[c])).sum(axis=1)
            log_lik[:, c] = -self.dim * 0.5 * np.log(self.vars[c]) - 0.5 * np.log(2 * np.pi) \
                            - 0.5 / self.vars[c] * distance
        return log_lik

    def do_m_step(self, X, z):
        comp_sum = z.sum(axis=0)
        self.weight = comp_sum / comp_sum.sum()
        self.means_ = z.T.dot(X) / comp_sum[:, np.newaxis]
        for c in range(self.n_components):
            distance = np.square((X - self.means_[c])).sum(axis=1)
            distance = distance.dot(z[:, c])
            self.vars[c] = (distance * 0.5 + self.var_b) / (0.5 * self.dim * comp_sum[c] + self.var_a - 1)

