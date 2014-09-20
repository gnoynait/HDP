import numpy as np
import random
from sklearn import mixture
from utils import load_word_vec, title_vec

rest = 'REST'
def init_cops(cop_list, dir):
    f = open(cop_list)
    cops = {}
    cops[rest] = open(dir + rest, 'w')
    for line in f:
        line = line.strip()
        cops[line] = open(dir + line, 'w')
    f.close()
    return cops

def url_to_cop(url, cops):
    path = []
    #try:
    surl = url.split('://')
    surl = surl[1].split('/')
    host = surl[0]
    dir = surl[1:]
    shost = host.split('.')
    shost.reverse()
    path = shost + dir
    #except:
    #    print url
    #    return rest
    for l in range(len(path), 0, -1):
        cop = '.'.join(path[:l])
        if cop in cops:
            return cop
    else:
        return rest

def process_dataset(dataset,cop_list = '/home/pawnty.ty/data/sina_prefix_list.dat', dir = '/home/pawnty.ty/data/cops/'):
    cops = init_cops(cop_list, dir)
    f = open(dataset)
    for i, line in enumerate(f):
        if i % 100000 == 0:
            print i / 100000
        line = line.replace(',', ' ', 1)
        record = line.split()
        if len(record) < 5:
            continue
        cop = url_to_cop(record[0], cops)
        cops[cop].write(' '.join(record[1:]) + '\n')
    for cop in cops.values():
        cop.close()
    f.close()

def gen_cop_data(input_file, word_vec, max_line = 500):
    f = open(input_file)
    data = []
    for i, line in enumerate(f):
        if i >= max_line:
            break
        line = line.strip()
        x = title_vec(line.split(), word_vec)
        data.append(x)
    X = np.array(data)
    f.close()
    return titles, 10 * X
def cops_file(cops_file = '/home/pawnty.ty/data/cops_file.dat'):
    f = open(cops_file)
    cops = []
    for line in f:
        cops.append(line.strip())
    f.close()
    return cops
def run_hdp():
    word_vec = load_word_vec()
    cops = cops_file()
    random.shuffle(cops)
    hdp = onlinehdpgmm.onlinehdp()
    for i in range(0, len(cops), 10):
        batch = []
        for c in cops[i:i+10]:
            x = preprocess(cops[i])
            batch.append(x)
        hdp.process_documents(batch)
    hdp.save_model()

