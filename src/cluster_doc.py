import sys
from collections import Counter
import cPickle

import numpy as np
import model as md

########################################################################################
#### base = md.FullFactorSpheGaussianMixture(500, dim, 1e-10, 1e10, 1e5,  X, sheduler)
########################################################################################
vect_file = 'D:\documents\gmm_experiments\data\\norm_words_vec.npy'
dim = 100
label_file = 'D:\documents\gmm_experiments\data\site.txt'

lines = open(label_file).readlines()
label_cnter = Counter(lines)
label = np.array(lines, dtype=np.str_)
X = np.load(vect_file)

instance_number = X.shape[0]
#instance_number = 50000
perm = np.arange(instance_number)
np.random.shuffle(perm)
X = X[perm,:]
label = label[perm]

sheduler = md.DecaySheduler(100, 0.6, 0.000001)

print label_cnter.most_common(300)
print len(label_cnter)
#sys.exit(0)
used_label = set(l for l, c in label_cnter.most_common(300))

groups = [X[label==l] for l in used_label]


belief = 1e10
variance = 1e-2
base = md.FullFactorSpheGaussianMixture(500, dim, 1e-10, belief, variance * belief,  X, sheduler)
base = md.DPMixture(500, 1, base, md.StickBreakingWeight(500, 1), md.DecaySheduler(100, 0.6, 0.000000001))

weight = md.StickBreakingWeight(200, 10)
#weight = md.DirichletWeight(200, 0.001)

models = [md.SubDPMixture(200, 10, base, md.DecaySheduler(1, 0.6, 0.000000001))]

#model = md.SubDPMixture(200, 1, base, sheduler)
#model = md.DPMixture(200, 1, base, weight, sheduler)

output = open('exp1.txt', 'w')
batch_size = 200

for belief, variance in [
    (1e5, 1e-2),
    (1e3, 1e-2),
    (1e1, 1e-2),
    ] * 3:
    base = md.FullFactorSpheGaussianMixture(500, dim, 1e-10, belief, variance * belief,  X, sheduler)
    base = md.DPMixture(500, 1, base, md.StickBreakingWeight(500, 1), md.DecaySheduler(100, 0.6, 0.000000001))
    for i in range((X.shape[0] + batch_size - 1) / batch_size):
        if (i + 1) % 100 == 0:
            print 'iteration: %d' % (i + 1)
        #batch = X[ np.random.choice(X.shape[0], batch_size, replace=True)]
        batch = X[i * batch_size:min((i+1) * batch_size, X.shape[0]), :]
        base.update(batch, 1e6)
    n = sum(np.exp(base.weight.logWeight())>=0.001)
    line = '{:e},{:e},{:d}\n'.format(belief, variance, n)
    print line
    tweight = np.exp(base.weight.logWeight())
    print sorted(tweight, reverse=True)
    output.write(line)
    output.flush()
sys.exit(0)
cPickle.dump(base, open('base_model.pkl', 'w'))

tweight = np.exp(base.weight.logWeight())
print sorted(tweight, reverse=True)
print sum(tweight), max(tweight), min(tweight)
print sum(tweight >= 0.001)
y = []
tt = (X.shape[0] + 99) / 100
tt = 100
for i in range(100):
    p = base.predict(X[i * 100:min((i+1) * 100, X.shape[0])])
    y.extend(list(p))

y_unique = np.unique(y)
print y_unique

sys.exit(0)

for i in range(100000):
    if (i + 1) % 100 == 0:
        print 'iteration: %d' % (i + 1)
    for data, model in zip(groups, models):
        batch = data[ np.random.choice(data.shape[0], batch_size, replace=True)]
        model.update(batch, 1e6)

y = []
for data, model in zip(groups, models):
    for i in range((data.shape[0] + 99) / 100):
        p = model.predict(data[i * 100:min((i+1) * 100, data.shape[0])])
        y.extend(list(p))

#np.save('y.npy', y)
y_unique = np.unique(y)
print y_unique