import re
import pickle
import codecs
import random
import numpy as np
import torch
import torch.nn as nn


def load_word2vec(emb_path, word_dim, num_char, char_dim):
    with open("data/maps.pkl", 'rb') as f:
        _, id_to_word, __, ___ = pickle.load(f)
    old_weights = nn.init.xavier_uniform_(torch.zeros(num_char, char_dim))
    new_weights = old_weights.numpy()
    pre_trained = {}
    emb_invalid = 0
    for i, line in enumerate(codecs.open(emb_path, 'r', 'utf-8')):
        line = line.rstrip().split()
        if len(line) == word_dim + 1:
            pre_trained[line[0]] = np.array([float(x) for x in line[1:]]).astype(np.float32)
        else:
            emb_invalid += 1
    c_found = 0
    c_lower = 0
    c_zeros = 0
    n_words = len(id_to_word)
    for i in range(n_words):
        word = id_to_word[i]
        if word in pre_trained:
            new_weights[i] = pre_trained[word]
            c_found += 1
        elif word.lower() in pre_trained:
            new_weights[i] = pre_trained[word.lower()]
            c_lower += 1
        elif re.sub('\d', '0', word.lower()) in pre_trained:
            new_weights[i] = pre_trained[re.sub('\d', '0', word.lower())]
            c_zeros += 1
    # weights = nn.Embedding(num_char, char_dim)
    # weights.weight.data.copy_(torch.from_numpy(new_weights))
    return new_weights

def load_tri_sentences(path):
    expand, sens, sen = list(), list(), list()
    c, l, t = list(), list(), list()
    for line in codecs.open(path, 'r', 'utf8'):
        line = line.rstrip()
        if line:
            word = line.split()
            c.append(int(word[0]))
            l.append(int(word[1]))
            t.append(int(word[2]))
        else:
            if len(c) > 0:
                sens.append([c, l, t])
                c, l, t = [], [], []
    for x in range(len(sens)):
        c, l, t = sens[x]        # one sentence
        for y in range(len(c)):
            if y > 0:
                mask = [1 for i in range(y)]            # 0 - (y-1)
                mask += [2 for i in range(len(c) - y)]  # total of (len(c)-y+y)=len(c) numbers
                cut = y
                tri_in = t[y]
                expand.append([c, tri_in, mask, cut])
    return expand

def load_arg_sentences(path):
    expand, sens, sen = list(), list(), list()
    c, l, t, a = list(), list(), list(), list()
    for line in codecs.open(path, 'r', 'utf8'):
        line = line.rstrip()
        if line:
            word = line.split()
            c.append(int(word[0]))
            l.append(int(word[1]))
            t.append(int(word[2]))
            a.append(int(word[3]))
        else:
            if len(c) > 0:
                sens.append([c, l, t, a])
                c, l, t, a = [], [], [], []
    for x in range(len(sens)):
        sen = sens[x]
        tri_f = 0
        for i in range(len(sen[0])):
            if sen[2][i] != 0:
                tri_f = i
                break
        for y in range(len(sen[0])):
            if y > 0 and y != tri_f:
                fir = min(tri_f, y)
                sec = max(tri_f, y)
                mask = [1 for i in range(fir)]
                mask += [2 for i in range(sec - fir)]
                mask += [3 for i in range(len(sen[0]) - sec)]
                cut = [tri_f, y]
                tri_loc, arg_loc = [], []
                for i in range(len(sen[0])):
                    tri_loc.append(i - cut[0])
                    arg_loc.append(i - cut[1])
                arg_in = [0 for i in range(36)]
                arg_in[sen[3][y]] = 1
                expand.append([sen[0], sen[1], sen[2], sen[3], arg_in, tri_loc, arg_loc, mask, cut])
    print("load_finished!")
    return expand


class Batch_tri(object):
    def __init__(self, data, batch_size, sen_len):
        self.batch_data = self.sort_pad(data, batch_size, sen_len)
        self.len_data = len(self.batch_data)
        self.length = int(sen_len)

    def sort_pad(self, data, batch_size, sen_len):
        num_batch = int(len(data)/batch_size)
        sort_data = sorted(data, key=lambda x: len(x[0]))
        batch_data = list()
        for i in range(num_batch):
            batch_data.append(self.pad(sort_data[i * batch_size: (i + 1) * batch_size], sen_len))
        return batch_data

    @staticmethod
    def pad(data, length):
        chars, tri_in, pf_in, mask, cut, lxl_in = list(), list(), list(), list(), list(), list()
        for line in data:
            c, t, m, cu = line
            padding = [0] * (length - len(c))
            chars.append(c + padding)
            tri_in.append(t)
            pf_in.append([(i - cu + length - 1) for i in range(length)])
            lxl_in.append((c + padding)[cu-1:cu+2])
            mask.append(m + [0] * (length - len(c) - 2))
            cut.append(cu)
        return [chars, tri_in, pf_in, lxl_in, mask, cut]

    def iter_batch(self):
        random.shuffle(self.batch_data)
        for i in range(self.len_data):
            yield self.batch_data[i]


class Batch_arg(object):
    def __init__(self, data, batch_size, sen_len):
        self.batch_data = self.sort_pad(data, batch_size, sen_len)
        self.len_data = len(self.batch_data)
        self.length = int(sen_len)

    def sort_pad(self, data, batch_size, sen_len):
        num_batch = int(len(data)/batch_size)
        sort_data = sorted(data, key=lambda x: len(x[0]))
        batch_data = list()
        for i in range(num_batch):
            batch_data.append(self.pad(sort_data[i * batch_size: (i + 1) * batch_size], sen_len))
        return batch_data

    @staticmethod
    def pad(data, length):
        chars, ls, tri, arg, arg_in, tri_loc, arg_loc, mask, cut = list(), list(), list(), list(), list(), list(), list(), list(), list()
        for line in data:
            c, l, t, a, a_i, t_l, a_l, m, cu = line
            padding = [0] * (length - len(c))
            chars.append(c + padding)
            ls.append(l + padding)
            tri.append(t + padding)
            arg.append(a + padding)
            arg_in.append(a_i)
            mask.append(m + [0] * (length - len(c) - 2))
            cut.append(cu)
            for i in range(length - len(c)):
                t_l.append(t_l[len(c) - 1] + i + 1)
                a_l.append(a_l[len(c) - 1] + i + 1)
            for i in range(len(c)):
                t_l[i] += length - 1
                a_l[i] += length - 1
            tri_loc.append(t_l)
            arg_loc.append(a_l)
        return [chars, ls, tri, arg, arg_in, tri_loc, arg_loc, mask, cut]

    def iter_batch(self):
        random.shuffle(self.batch_data)
        for i in range(self.len_data):
            yield self.batch_data[i]