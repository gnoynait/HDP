import numpy as np
import matplotlib.pyplot as plt
from numpy.random import multivariate_normal as normal
from numpy.random import gamma
from numpy.random import dirichlet
from numpy.random import multinomial
import onlinedpgmm

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
    T = 40
    K = 40 
    D = 100
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
    groups = map(lambda m, w: onlinedpgmm.Group(alpha, 1000, onlinedpgmm.RandomGaussMixtureData(w, m, cov)),mean, weight)
                        
    for i in range(D):
        hdp.process_groups(groups)
    for group in groups:
        group.report()

test_hdp_2()
