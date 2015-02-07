import numpy as np
from collections import defaultdict
import sys
from onlinedpgmm import *
import os
import random
import sys
from sklearn import metrics

def load_data(fname):
    group = defaultdict(list)
    with open(fname) as f:
        for line in f:
            record = line.strip().split()
            x = [float(a) for a in record[1:]]
            group[record[0]].append(x)
    return group.values()

def hdpcluster(fname):
    T = 100
    K = 10 
    gamma = 5
    alpha = 1
    kappa = 0.6
    tau = 1
    total = 2500000
    dim = 20
    mode = 'semi-spherical'
    epoch = 100
    batchsize = 20 
    #batchgroup = 3

    data = [ListData(d) for d in load_data(fname)]
    sample = []
    for d in data:
        X = d.sample(int(2 * T/len(data)))
        sample.append(X)
    sample = np.vstack(sample)
    np.random.shuffle(sample)
    groups = [Group(alpha, K, T, d.size(), batchsize, d) for d in data]
    hdp = OnlineHDP(T, K, alpha, gamma, kappa, tau, total, dim, mode)
    hdp.init_par(init_mean = sample[:T,:], init_cov=0.001, \
        prior_x0=(1.0, 1000.0))
    del sample
    for i in range(epoch):
        print '\rprocess %d out of %d' % (i, epoch),
        hdp.process_groups(groups)
    print '\nfinished'

    labels_true = []
    labels_pred = []
    for l, group in enumerate(groups):
        X = group.data.X
        Y = hdp.predict(X, group).tolist()
        labels_true.extend([l] * len(X))
        labels_pred.extend(Y)
    return labels_true, labels_pred


if __name__ == '__main__':
    fname = sys.argv[1]
    labels_true, labels_pred = hdpcluster(fname)
    print 'CN:\t',  len(set(labels_pred))
    print 'ARI:\t', metrics.adjusted_rand_score(labels_true, labels_pred)
    #print 'MI:\t', metrics.mutual_info_score(labels_true, labels_pred)
    print 'AMI:\t', metrics.adjusted_mutual_info_score(labels_true, labels_pred)
    print 'NMI:\t', metrics.normalized_mutual_info_score(labels_true, labels_pred)
    #print 'HOM:\t', metrics.homogeneity_score(labels_true, labels_pred)
    #print 'COM:\t', metrics.completeness_score(labels_true, labels_pred)
    print 'VME:\t', metrics.v_measure_score(labels_true, labels_pred)
