import numpy as np
import matplotlib.pyplot as plt
from numpy.random import multivariate_normal as normal
from numpy.random import gamma
from numpy.random import dirichlet
from numpy.random import multinomial
import onlinedpgmm
import itertools
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

import time

#random_seed = 999931111
random_seed = int(time.time())
np.random.seed(random_seed)
def gen_parameter(dim, k):
    means = normal(np.zeros(dim), np.diag(np.ones(dim) * 10), k)
    #precis = gamma(1, 1, k)
    precis = np.ones(k)
    return means, precis

def gen_data(means, precis, n):
    weight = dirichlet(np.ones(means.shape[0]))
    count = multinomial(n, weight)
    data = np.zeros((n, means.shape[1]))
    start = 0
    for i in range(len(count)):
        data[start: start + count[i], :] = normal(means[i], np.diag(precis[i] * np.ones(means.shape[1])), count[i])
        start = start + count[i]
    s = np.arange(n)
    np.random.shuffle(s)
    return data[s]

def gen_cops(means, precis, batch_size, cop_size):
    cops = []
    for i in range(batch_size):
        cops.append(gen_data(means, precis, cop_size))
    return cops

def test_hdp_1():
    T = 100
    K = 40 
    topics = 10
    D = 100
    alpha = 2 
    gamma = 0.1 
    kappa = 0.9
    tau = 1
    dim = 200
    total = 500000
    mode = 'diagonal'
    #mode = 'full'
    hdp = onlinedpgmm.online_hdp(T, K, D, alpha, gamma, kappa, tau, total, dim, mode)
    var_converge = 0.00001

    batch_size = 1
    means, precis = gen_parameter(dim, topics)
    data = gen_data(means, precis, cop_size)
    plt.scatter(data[:, 0], data[:, 1], marker = '.')
    hdp.new_init(data)
    init_means = hdp.m_means
    plt.scatter(init_means[:, 0], init_means[:, 1], c = 'y')
    #data = gen_cops(means, precis, batch_size, cop_size) 
    for i in range(D):
        data = gen_cops(means, precis, batch_size, cop_size) 
        hdp.process_documents(data, var_converge)
        #hdp.process_documents(data, var_converge)
    model = open('model.dat', 'w')
    weight = np.exp(onlinehdpgmm.expect_log_sticks(hdp.m_var_sticks))
    thresh = 0.02
    infer_means = hdp.m_means[weight > thresh]
    plt.scatter(means[:,0], means[:,1], c = 'g', marker='>')
    plt.scatter(infer_means[:, 0], infer_means[:, 1], c = 'r')
    plt.show()
    print hdp.m_precis[weight > thresh]
    hdp.save_model(model)
    model.close()

def test_hdp_2():
    T = 20
    K = 20 
    D = 1000
    alpha = 0.5 
    gamma = 1 
    kappa = 0.9
    tau = 1
    dim = 2
    total = 500000
    mode = 'diagonal'
    #mode = 'full'
    hdp = onlinedpgmm.online_hdp(T, K, D, alpha, gamma, kappa, tau, total, dim, mode)
    var_converge = 0.00001

    mean = np.array(  [[0.0, 0.0],
                        [3.0, 3.0],
                        [-3.0, 3.0],
                        [-3.0, -3.0],
                        [3.0, -3.0]])
    cov = np.zeros((5, 2, 2))
    cov += np.array(    [[1.0, 0.0],
                         [0.0, 2.0]])
    weight = np.array([[0.2, 0.2, 0.6, 0.0, 0.0],
                       [0.0, 0.5, 0.0, 0.5, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 1.0],
                       [0.3, 0.1, 0.3, 0.2, 0.1],
                       [0.1, 0.0, 0.2, 0.3, 0.4]])
    groups = map(lambda w: onlinedpgmm.Group(alpha, 1000, onlinedpgmm.RandomGaussMixtureData(w, mean, cov)), weight)
                        
    for i in range(D):
        hdp.process_groups(groups)
    for group in groups:
        group.report()

def plot(axis, model, X, Y_, title, lim = None, show = 'md'):
    color_iter = itertools.cycle(['r', 'g', 'b', 'c', 'm', 'y', 'k'])
    Y_ = model.predict(X)
    for i, (mean, cov, col) in enumerate(zip(
            model.m_means, model.get_cov(), color_iter)):
        v, w = linalg.eigh(cov)
        u = w[0] / linalg.norm(w[0])
        if not np.any(Y_ == i):
            continue
        if 'd' in show:
            axis.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=col)
        if 'm' in show:
            angle = np.arctan(u[1] / (u[0] + np.finfo(np.float32).eps))
            angle = 180 * angle / np.pi  # convert to degrees
            ell = mpl.patches.Ellipse(mean, v[0], v[1], 180 + angle, color=col)
            ell.set_clip_box(axis.bbox)
            ell.set_alpha(0.5)
            axis.add_artist(ell)
    if lim is not None:
        plt.xlim(lim[0], lim[1])
        plt.ylim(lim[2], lim[3])
    plt.title(title)
    plt.xticks(())
    plt.yticks(())

def show_grid():
    np.random.seed(random_seed)
    T = 20
    K = 20 
    D = 100000
    batch_size = 100
    process_round = 50

    gamma = 0.5 
    alpha = 0.5 
    kappa = 0.7
    tau = 1
    dim = 2
    var_converge = 0.00001
    total = 5000000
    mode = 'spherical'
    #mode = 'diagonal'
    #mode = 'full'
    hdp = onlinedpgmm.online_hdp(T, K, D, alpha, gamma, kappa, tau, total, dim, mode)

    mean = np.array(  [ [4.0, 4.0],
                        [-4.0, 4.0],
                        [-4.0, -4.0],
                        [4.0, -4.0]])
    cov = np.zeros((4, 2, 2))
    cov += np.array(    [[1.0, 0.0],
                         [0.0, 1.0]])
    weight = np.array([[0.2, 0.2, 0.6, 0.0, 0.0],
                       [0.0, 0.5, 0.0, 0.5, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 1.0],
                       [0.3, 0.1, 0.3, 0.2, 0.1],
                       [0.1, 0.0, 0.2, 0.3, 0.4]])
    weight = np.array([ [0.5, 0.5, 0.0, 0.0],
                        [0.5, 0.0, 0.5, 0.0],
                        [0.5, 0.0, 0.0, 0.5],
                        [0.0, 0.5, 0.5, 0.0],
                        [0.0, 0.5, 0.0, 0.5],
                        [0.0, 0.0, 0.5, 0.5]])
                        #[0.25, 0.25, 0.25, 0.25]])
    groups = map(lambda w: onlinedpgmm.Group(alpha, 1000, onlinedpgmm.RandomGaussMixtureData(w, mean, cov)), weight)
                        
    for i in range(process_round):
        hdp.process_groups(groups)
    X = []
    Y = []
    for g in groups:
        sample = g.data.sample(200)
        X.append(sample)
        Y.append(hdp.predict(sample, group = g))
    X = np.vstack(X)
    Y = np.vstack(Y)
    lim = [-10, 10, -10, 10]
    #plot(plt.subplot(221), dp, X, 'DP', lim, 'd')
    #plot(plt.subplot(222), dp, X, 'DP', lim, 'm')
    plot(plt.subplot(223), hdp, X, Y, 'HDP', lim, 'd')
    plot(plt.subplot(224), hdp, X, Y, 'HDP', lim, 'm')
    plt.show()
show_grid()
