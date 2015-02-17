import numpy as np
from collections import defaultdict
from collections import Counter
import sys
from onlinedpgmm import *
import os
import random
import sys
from sklearn import metrics

def split(record, p):
    size = (len(record) + p - 1) / p
    res = []
    for i in range(0, len(record), size):
        res.append(record[i: min(i + size, len(record))])
    return res

def load_data(fname, p, q):
    klass = defaultdict(list)
    with open(fname) as f:
        for line in f:
            record = line.strip().split()
            k = record[0]
            x = [float(a) for a in record[1:]]
            klass[k].append((k, x))
    trunk = []
    for records in klass.values():
        trunk.extend(split(records, p))
    random.shuffle(trunk)
    group = []
    label = []
    while len(trunk) > 0:
        g = []
        for i in range(min(q, len(trunk))):
            g.extend(trunk.pop())
        group.append([r[1] for r in g])
        label.extend([r[0] for r in g])
    return label, group

def hdpcluster(fname,dim, p, q):
    #p  split each class to p trunks
    #q  combine q trunks to a new group
    T = 100 
    K = 10
    gamma = 1
    alpha = 1
    kappa = 0.6
    tau = 1
    total = 100000
    mode = 'semi-spherical'
    epoch = 100 #control number of iteration
    batchsize = 2 # control #samples in a batch
    batchgroup = 3 # control #groups processed in one iteration

    labels_true, data = load_data(fname, p, q)
    data = [ListData(d) for d in data]
    sample = []
    for d in data:
        sample.append(d.X)
    sample = np.vstack(sample)
    np.random.shuffle(sample)
    groups = [Group(alpha, K, T, d.size(), batchsize, d) for d in data]
    hdp = OnlineHDP(T, K, alpha, gamma, kappa, tau, total, dim, mode)
    hdp.init_par(init_mean = sample[:T,:], init_cov=0.001, \
        prior_x0=(1.0, 1000.0))
    del sample
    for i in range(epoch):
        print '\rprocess %d out of %d' % (i, epoch),
        #hdp.process_groups(random.sample(groups, batchgroup))
        hdp.process_groups(groups)
    print

    labels_pred = []
    for group in groups:
        X = group.data.X
        Y = hdp.predict(X, group).tolist()
        labels_pred.extend(Y)
    return labels_true, labels_pred


if __name__ == '__main__':
    fname = sys.argv[1]
    dim = int(sys.argv[2])
    p = int(sys.argv[3])
    q = int(sys.argv[4])
    labels_true, labels_pred = hdpcluster(fname, dim, p, q)
    print 'CN:\t',  len(set(labels_pred)), len(set(labels_true))
    print 'ARI:\t', metrics.adjusted_rand_score(labels_true, labels_pred)
    #print 'MI:\t', metrics.mutual_info_score(labels_true, labels_pred)
    print 'AMI:\t', metrics.adjusted_mutual_info_score(labels_true, labels_pred)
    print 'NMI:\t', metrics.normalized_mutual_info_score(labels_true, labels_pred)
    print 'HOM:\t', metrics.homogeneity_score(labels_true, labels_pred)
    print 'COM:\t', metrics.completeness_score(labels_true, labels_pred)
    #print 'VME:\t', metrics.v_measure_score(labels_true, labels_pred)
