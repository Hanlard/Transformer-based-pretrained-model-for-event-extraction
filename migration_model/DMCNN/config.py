import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from sklearn.metrics import f1_score, precision_score, recall_score

from migration_model.DMCNN.dmcnn import dmcnn_t
from migration_model.DMCNN.loader import load_word2vec
from migration_model.DMCNN.loader import Batch_tri, Batch_arg
from migration_model.DMCNN.loader import load_tri_sentences, load_arg_sentences


def to_var(x):
    return Variable(torch.from_numpy(x).long().cuda())


class Config(object):
    def __init__(self):
        self.gpu = "4"
        self.path_t = 'data/tri.train'
        self.path_a = 'data/arg.train'
        self.path_test_t = 'data/tri.test'
        self.path_test_a = 'data/arg.test'
        self.path_modelt = 'data/modelt'
        self.path_debug = 'data/debug'
        self.lr = 1
        self.weight_decay = 1e-5
        self.epoch = 75                 # training epoches
        self.epoch_save = 1
        self.sen = 80                   # sentence length
        self.char_dim = 100             # length of word embedding tensor
        self.num_char = 20136           # total num of word2vec model
        self.batch_t = 170              # num of sentences in one batch
        self.batch_a = 20
        self.num_t = 34                 # num of triggers
        self.num_a = 36
        self.pf_t = 5                   # dim of pf in trigger classification
        self.pf_a = 5
        self.ef_a = 5
        self.window_t = 3               # window size in cnn
        self.window_a = 3
        self.feature_t = 200            # num of features in cnn
        self.feature_a = 300

    def load_traint_data(self):
        print("Reading training data...")
        train_t = load_tri_sentences(self.path_t)
        self.train_t_b = Batch_tri(train_t, self.batch_t, self.sen)
        self.emb_weights = load_word2vec("data/100.utf8", 100, self.num_char, self.char_dim)
        print("finish reading")

    def load_testt_data(self):
        print("Reading testing data...")
        test_t = load_tri_sentences(self.path_test_t)
        self.test_t_b = Batch_tri(test_t, self.batch_t, self.sen)
        print("finish reading")


    def set_traint_model(self):
        print("Initializing training model...")
        self.modelt = dmcnn_t(config=self)
        self.optimizer_t = optim.Adadelta(self.modelt.parameters(), lr=self.lr, rho=0.95, eps=1e-6, weight_decay=self.weight_decay)
        self.modelt.cuda()
        for param_tensor in self.modelt.state_dict():
            print(param_tensor, "\t", self.modelt.state_dict()[param_tensor].size())
        print("Finish initializing")

    def set_testt_model(self):
        print("Initializing testing model...")
        self.model_test_t = dmcnn_t(config=self)
        self.model_test_t.cuda()
        self.model_test_t.eval()
        print("finish initializing")

    def train_one_step(self, batch):
        self.modelt.char_inputs = to_var(np.array(batch[0]))
        self.modelt.trigger_inputs = to_var(np.array(batch[1]))
        self.modelt.pf_inputs = to_var(np.array(batch[2]))
        self.modelt.lxl_inputs = to_var(np.array(batch[3]))
        self.modelt.masks = to_var(np.array(batch[4]))
        self.modelt.cuts = to_var(np.array(batch[5]))

        self.optimizer_t.zero_grad()
        loss, maxes= self.modelt()
        loss.backward()
        self.optimizer_t.step()
        return loss.data, maxes

    def test_one_step(self, batch):
        self.model_test_t.char_inputs = to_var(np.array(batch[0]))
        self.model_test_t.trigger_inputs = to_var(np.array(batch[1]))
        self.model_test_t.pf_inputs = to_var(np.array(batch[2]))
        self.model_test_t.lxl_inputs = to_var(np.array(batch[3]))
        self.model_test_t.masks = to_var(np.array(batch[4]))
        self.model_test_t.cuts = to_var(np.array(batch[5]))

        loss, maxes = self.model_test_t()
        return loss, maxes

    def train(self):
        for epoch in range(self.epoch):
            losses = 0
            tru = pre = None
            i = 0
            print("epoch: ", epoch)
            for batch in self.train_t_b.iter_batch():
                loss, maxes = self.train_one_step(batch)
                losses += loss
                if i == 0:
                    tru = self.modelt.trigger_inputs
                    pre = maxes
                else:
                    tru = torch.cat((tru, self.modelt.trigger_inputs), dim=0)
                    pre = torch.cat((pre, maxes), dim=0)
                i += 1
            tru = tru.cpu()
            pre = pre.cpu()
            prec = precision_score(tru, pre, labels=list(range(1, 34)), average='micro')
            rec = recall_score(tru, pre, labels=list(range(1, 34)), average='micro')
            f1 = f1_score(tru, pre, labels=list(range(1, 34)), average='micro')
            i = 0
            if epoch % self.epoch_save == 0:
                torch.save(self.modelt.state_dict(), self.path_modelt)
            print("loss_average:", losses/i)
            print("Precision:  ", prec)
            print("Recall:  ", rec)
            print("FMeasure", f1)


    def test(self):
        self.model_test_t.load_state_dict(torch.load(self.path_modelt))
        tru = pre = None
        i = 0
        with open(self.path_debug, 'w') as f:
            losses = 0
            for batch in self.test_t_b.iter_batch():
                loss, maxes = self.test_one_step(batch)
                losses += loss
                if i == 0:
                    tru = self.model_test_t.trigger_inputs
                    pre = maxes
                else:
                    tru = torch.cat((tru, self.model_test_t.trigger_inputs), dim=0)
                    pre = torch.cat((pre, maxes), dim=0)
                i += 1
            tru = tru.cpu()
            pre = pre.cpu()
            tru_n = tru.numpy()
            pre_n = pre.numpy()
            for p in range(self.batch_t*self.test_t_b.len_data):
                if tru_n[p] != pre_n[p]:
                    f.write(str(tru_n[p]) + ':' + str(pre_n[p]) + '\n')
            prec = precision_score(tru, pre, labels=list(range(1, 34)), average='micro')
            rec = recall_score(tru, pre, labels=list(range(1, 34)), average='micro')
            f1 = f1_score(tru, pre, labels=list(range(1, 34)), average='micro')
            print("loss_average:  ", losses/i)
            print("Precision:  ", prec)
            print("Recall:  ", rec)
            print("FMeasure", f1)



