import numpy as np
import sys
from onlinedpgmm import *
import os
import random
def rong_parser(line):
    record = line.split('##')
    return [float(a) for a in record[2].split()]

def hdpcluster():
    T = 30
    K = 10
    gamma = 1
    alpha = 1
    kappa = 0.6
    tau = 1
    total = 250000000
    dim = 300
    mode = 'semi-spherical'
    epoch = 500
    batchsize = 20 
    #batchgroup = 3

    input_dir = '/home/pawnty/data/groups/'
    output_file = '/home/pawnty/data/cluster.txt'

    hdp = OnlineHDP(T, K, alpha, gamma, kappa, tau, total, dim, mode)
    hdp.init_par(init_cov=0.001, prior_x0=(1.0, 1000.0))

    input_files = [input_dir + f for f in os.listdir(input_dir)]
    data = [FileData(f, rong_parser) for f in input_files]
    groups = [Group(alpha, K, T, d.size(), batchsize, d) for d in data]
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
        topic, title, X = group.data.next_n_record(1000000)
        Y = hdp.predict(X, group, 2).tolist()
        #Y = hdp.predict(X).tolist()
        titles.extend(title)
        topics.extend(topic)
        Ys.extend(Y)
    result = list(zip(Ys, topics, titles))
    result.sort()
    with open(output_file, 'w') as outfile:
        for r in result:
            outfile.write('%d\t%s\t%s\n' % r)
if __name__ == '__main__':
    hdpcluster()
