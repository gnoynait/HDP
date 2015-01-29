import numpy as np
import sys
from onlinedpgmm import *
import os
import random
def rong_parser(line):
    record = line.split('##')
    return [float(a) for a in record[2].split()]

def hdpcluster():
    T = 100
    K = 20
    gamma = 1
    alpha = 0.1
    kappa = 0.6
    tau = 1
    total = 200000000000
    dim = 300
    mode = 'semi-spherical'
    epoch = 500
    batchsize = 20 
    batchgroup = 3
    size = 10000

    input_dir = '/home/pawnty/data/groups/'
    output_file = '/home/pawnty/data/cluster.txt'

    hdp = OnlineHDP(T, K, alpha, gamma, kappa, tau, total, dim, mode)
    hdp.init_par(init_cov=0.001, prior_x0=(1.0, 10000.0))

    input_files = [input_dir + f for f in os.listdir(input_dir)]
    data = [FileData(f, rong_parser) for f in input_files]
    groups = [Group(alpha, K, size, batchsize, d) for d in data]
    for i in range(epoch):
        print 'process %d out of %d' % (i, epoch)
        #hdp.process_groups(random.sample(groups, batchgroup))
        hdp.process_groups(groups)

    titles = []
    topics = []
    X = []
    Ys = []
    for group in groups:
        group.data.reset()
        topic, title, X = group.data.next_n_record(100)
        Y = hdp.predict(X, group).tolist()
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
