import numpy as np

def prefix_to_url(prefix):
    path = prefix.split('./')
    domain = path[0].split('.')
    domain.reverse()
    url = ['http://', '.'.join(domain)] +  ['/' + p for p in path[1:]]
    return ''.join(url)

def load_word_vec(vec_file = '/home/pawnty.ty/data/raw/all_vec_text'):
    word_vec = {}
    f = open(vec_file)
    for line in f:
        row = line.split()
        if len(row) < 10:
            continue
        word_vec[row[0]] = np.array(row[1:], dtype='float')
    return word_vec

def vec_sim(vec1, vec2):
    p = vec1.dot(vec2)
    r1 = np.sqrt(np.square(vec1).sum())
    r2 = np.sqrt(np.square(vec2).sum())
    return p / r1 / r2

def search_sim_words(vec, word_vec, thresh = 0.5):
    words = []
    for w in word_vec:
        s = vec_sim(word_vec[w], vec)
        if s > thresh:
            words.append((s, w))
    words.sort(reverse = True)
    return words

def vec_average(vec_list):
    if len(vec_list) == 0:
        return 0
    a = np.zeros(vec_list[0].size)
    for v in vec_list:
        a = a + v
    return a / len(vec_list)

def repeat_search(vec, word_vec, times):
    words = None
    for t in range(times):
        words = search_sim_words(vec, word_vec)
        if (len(words) < 5):
            break
        vec = vec_average([word_vec[w] for (s, w) in words[:50]])
    return words, vec
def normalize(vec):
    a = np.sqrt(np.sum(vec.dot(vec)))
    if a == 0:
        return normalize(np.ones(len(vec)))
    return vec / a

def exp_title_topic(title_vecs, topics):
    max_sim = 0
    topic = None
    for t, v in topics.iteritems():
        sim = 0
        for w in title_vecs:
            sim += np.exp(w.dot(v))
        if sim > max_sim:
            max_sim = sim
            topic = t
    return topic

def load_topic(word_vec, topic_file = '/home/pawnty.ty/data/raw/topic_words'):
    f = open(topic_file)
    topics = {}
    for line in f:
        row = line.split(',')
        words = row[1].split()
        vec = []
        for w in words:
            vec.append(word_vec[w])
        topic_vec = vec_average(vec)
        topics[row[0]] = normalize(topic_vec)
    f.close()
    return topics

def static_sim(vec, word_vec):
    dist = [0] * 21
    for k, v in word_vec.iteritems():
        sim = vec_sim(vec, v)
        i = int((sim + 1) * 10)
        dist[i] += 1
    return dist

def title_vec(title, word_vec):
    vec = np.zeros(500)
    for w in title:
        if w not in word_vec:
            continue
        else:
            vec += word_vec[w]
    return normalize(vec)

def url_prefix_train(train_file, word_vec):
    f = open(train_file)
    prefix_vec = {}
    prefix_count = {}
    for line in f:
        row = line.split()
        prefix = row[0]
        words = row[1:]
        tvec = title_vec(words, word_vec)
        if tvec == None:
            continue
        if prefix not in prefix_vec:
            prefix_vec[prefix] = tvec
            prefix_count[prefix] = 0
        else:
            prefix_vec[prefix] += tvec
            prefix_count[prefix] += 1
    f.close()
    for p in prefix_vec:
        prefix_vec[p] /= prefix_count[p]
    return prefix_vec

def url_prefix_read(read_file):
    prefix_vector = {}
    f = open(read_file)
    for line in f:
        row = line.split()
        prefix_vector[row[0]] = np.array(row[1:], dtype = 'float')
    f.close()
    return prefix_vector

def url_prefix_predict(title_vec, prefix_vec, threshold = 0.2):
    if title_vec == None:
        return None
    prefix = None
    projection = -100
    for p, v in prefix_vec.iteritems():
        pro = v.dot(title_vec)
        if pro > projection:
            projection = pro
            prefix = p
    return prefix

def url_prefix_cosine_predict(title_vec, prefix_vec, threshold = 0.2):
    if title_vec == None:
        return None
    prefix = None
    sim = -1
    for p, v in prefix_vec.iteritems():
        s = vec_sim(v, title_vec)
        if s > sim and s > threshold:
            sim = s
            prefix = p
    return prefix


def url_prefix_test(test_file, prefix_vec, word_vec):
    f = open(test_file)
    total = 0
    ignore = 0
    correct = 0
    for line in f:
        total += 1
        row = line.split()
        answer = row[0]
        words = row[1:]
        if len(words) == 0:
            ignore += 1
            continue
        tvec = title_vec(words, word_vec)
        result = url_prefix_predict(tvec, prefix_vec)
        print result
        if answer == result:
            correct += 1
    return float(correct) / (total - ignore)

def url_prefix_cosine_test(test_file, prefix_vec, word_vec, threshold):
    f = open(test_file)
    total = 0
    ignore = 0
    correct = 0
    for line in f:
        total += 1
        row = line.split()
        answer = row[0]
        words = row[1:]
        if len(words) == 0:
            ignore += 1
            continue
        tvec = title_vec(words, word_vec)
        result = url_prefix_cosine_predict(tvec, prefix_vec, threshold)
        print result
        if answer == result:
            correct += 1
    return float(correct) / (total - ignore)

