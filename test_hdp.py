import numpy as np
import matplotlib.pyplot as plt
from numpy.random import multivariate_normal as normal
from numpy.random import gamma
from numpy.random import dirichlet
from numpy.random import multinomial

import onlinehdpgmm
def gen_parameter(dim, k):
    means = normal(np.zeros(dim), np.diag(np.ones(dim)), k)
    precis = gamma(1, 1, k)
    return 10 * means, precis

def gen_data(means, precis, n):
    weight = dirichlet(np.ones(means.shape[0]))
    count = multinomial(n, weight)
    data = np.zeros((n, means.shape[1]))
    start = 0
    for i in range(len(count)):
        data[start: start + count[i], :] = normal(means[i], np.diag(precis[i] * np.ones(means.shape[1])), count[i])
        start = start + count[i]
    return data

def gen_cops(means, precis, batch_size, cop_size):
    cops = []
    for i in range(batch_size):
        cops.append(gen_data(means, precis, cop_size))
    return cops

def test_hdp():
    T = 10
    K = 10
    topics = 6 
    D = 50 
    alpha = 10 
    gamma = 20
    kappa = 0.7
    tau = 5
    dim = 2
    hdp = onlinehdpgmm.online_hdp(T, K, D, alpha, gamma, kappa, tau, dim)
    var_converge = 0.00001

    cop_size = 1000
    batch_size = 10
    means, precis = gen_parameter(dim, topics)
    #data = gen_data(means, precis, cop_size * 10)
    #hdp.new_init(data)
    for i in range(D):
        data = gen_cops(means, precis, batch_size, cop_size) 
        hdp.process_documents(data, var_converge)
    model = open('model.dat', 'w')
    hdp.save_model(model)
    model.close()

def test():
    means, precis = gen_parameter(2, 10)
    data = gen_data(means, precis, 1000) 
    return data

test_hdp()
