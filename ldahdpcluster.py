import numpy as np
import sys
from onlinedpgmm import *
import os
import random
def rong_parser(line):
    record = line.split('##')
    return [float(a) for a in record[2].split()]

def tain_parser(line):
    record = line.split()
    return [float(a) for a in record[1:]]

def hdpcluster():
    T = 100
    K = 10 
    gamma = 5
    alpha = 1
    kappa = 0.6
    tau = 1
    total = 250000000000
    dim = 20
    mode = 'semi-spherical'
    epoch = 500
    batchsize = 20 
    #batchgroup = 3

    input_dir = '/home/pawnty/data/groups/'
    output_file = '/home/pawnty/data/cluster.txt'


    input_files = [input_dir + f for f in os.listdir(input_dir)]
    data = [FileData(f, tain_parser) for f in input_files]
    sample = []
    for d in data:
        t, X = d.next_n_record(500)
        sample.append(X)
    sample = np.vstack(sample)
    np.random.shuffle(sample)
    groups = [Group(alpha, K, T, d.size(), batchsize, d) for d in data]
    hdp = OnlineHDP(T, K, alpha, gamma, kappa, tau, total, dim, mode)
    hdp.init_par(init_mean = sample[:T,:], init_cov=0.001, \
        prior_x0=(1.0, 1000.0))
    for i in range(epoch):
        print '\rprocess %d out of %d' % (i, epoch),
        #hdp.process_groups(random.sample(groups, batchgroup))
        hdp.process_groups(groups)

    titles = []
    topics = []
    X = []
    Ys = []
    for group in groups:
        group.data.reset()
        topic, X = group.data.next_n_record(1000000)
        Y = hdp.predict(X, group).tolist()
        #Y = hdp.predict(X).tolist()
        topics.extend(topic)
        Ys.extend(Y)
    result = list(zip(Ys, topics))
    result.sort()
    with open(output_file, 'w') as outfile:
        for r in result:
            outfile.write('%d\t%s\n' % r)
if __name__ == '__main__':
    hdpcluster()
