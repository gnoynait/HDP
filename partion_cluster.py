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
    res = []
    for i in range(0, len(record), (len(record) + p - 1) / p):
        res.append(record[i: min(i + p, len(record))])
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

def hdpcluster(fname):
    T = 200
    K = 3 
    gamma = 5
    alpha = 1
    kappa = 0.6
    tau = 1
    total = 2500000
    dim = 300
    mode = 'semi-spherical'
    epoch = 50 #control number of iteration
    batchsize = 2 # control #samples in a batch
    batchgroup = 3 # control #groups processed in one iteration
    p = 2 # split each class to p trunks
    q = 2 # combine q trunks to a new group

    labels_true, data = load_data(fname, p, q)
    data = [ListData(d) for d in data]
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
        #hdp.process_groups(random.sample(groups, batchgroup))
        hdp.process_groups(groups)
    print '\nfinished'

    labels_pred = []
    for group in groups:
        X = group.data.X
        Y = hdp.predict(X, group).tolist()
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
