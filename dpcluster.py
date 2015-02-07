import numpy as np
from onlinedpgmm import *
print "shouldn't use this"
def rong_parser(line):
    record = line.split('##')
    return [float(a) for a in record[2].split()]

def dpcluster():
    T = 100
    gamma = 1 
    kappa = 0.6
    tau = 1
    total = 7800
    dim = 300
    mode = 'semi-spherical'
    epoch = 400
    batchsize = 100
    input_file = '/home/pawnty/data/random_1_naive.txt'
    test_file = '/home/pawnty/data/random_1_naive.txt'
    output_file = '/home/pawnty/data/cluster.txt'

    dp = OnlineDP(T, gamma, kappa, tau, total, dim, mode)
    dp.init_par(init_cov=0.004, prior_x0=(1, 100))
    data = FileData(input_file, rong_parser)
    for i in range(epoch):
        print 'process %d out of %d' % (i, epoch)
        dp.process_documents([data.sample(batchsize)])
    title = []
    topic = []
    X = []
    with open(test_file) as infile:
        for line in infile:
            record = line.split('##')
            topic.append(record[0])
            title.append(record[1])
            X.append([float(a) for a in record[2].split()])
    X = np.array(X)
    Y = dp.predict(X).tolist()
    result = list(zip(Y, topic, title))
    result.sort()
    with open(output_file, 'w') as outfile:
        for r in result:
            outfile.write('%d\t%s\t%s\n' % r)
if __name__ == '__main__':
    dpcluster()
