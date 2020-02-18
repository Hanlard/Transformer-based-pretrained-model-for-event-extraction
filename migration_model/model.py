"""
2020/1/29修改
加入了 tri和arg双CRF
logits计算加入了滑动平均
"""
import os
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np

# from pytorch_pretrained_bert import BertModel
from transformers import BertModel

from DataLoadAndTrain import idx2trigger, argument2idx

from consts import NONE
from utils import find_triggers, find_argument
from migration_model.models.CRF import CRF
import copy

import numpy
import torch
from torch import nn
from torch.nn import functional as F
### 迁移 JMEE ###

from migration_model.enet.models.EmbeddingLayer import EmbeddingLayer, MultiLabelEmbeddingLayer

# from migration_model.enet.models.GCN import GraphConvolution
# from migration_model.enet.models.HighWay import HighWay
# from migration_model.enet.models.SelfAttention import AttentionLayer
#
# from migration_model.enet.util import BottledXavierLinear
#
# from migration_model.text_cls_models.TextCNN import NgramCNN



class Net(nn.Module):
    def __init__(self, trigger_size=None, entity_size=None, all_postags=None, PreModel=None, hyper_para =None, postag_embedding_dim=50,
                 argument_size=None, entity_embedding_dim=50, device=torch.device("cuda")):
        super().__init__()
        ## PreTrainModel

        self.PreModel = PreModel
        self.hp = hyper_para
        self.tri_embsize = 100
        self.pos_embsize = 100
        self.arg_embsize = 100

        self.trigger_embed = nn.Embedding(num_embeddings=trigger_size, embedding_dim=self.tri_embsize)
        self.argument_embed = nn.Embedding(num_embeddings=argument_size, embedding_dim=self.arg_embsize)
        self.position_embed = nn.Embedding(num_embeddings=100, embedding_dim=self.pos_embsize)

        self.argument_size = argument_size

        self.hidden_size = self.bert.config.hidden_size

        self.tri_fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.hidden_size, trigger_size + 2, bias=True),
        )

        self.tri_fc2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.hidden_size + self.tri_embsize, trigger_size + 2, bias=True),
        )

        self.arg_fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.hidden_size*2+self.tri_embsize*2+self.pos_embsize, argument_size + 2, bias=True),
        )

        self.arg_fc2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.hidden_size*2+self.tri_embsize*2+self.pos_embsize+ self.arg_embsize, argument_size + 2, bias=True),
        )
        # 自注意力
        # self.selfATT = AttentionLayer(D=self.hidden_size, H=512, return_sequences=False)

        # fc_trigger
        # self.fc_trigger = BottledXavierLinear(in_features=self.hidden_size*2, out_features=trigger_size).to(
        #     device=device)

        # fc_argument
        # self.fc_argument = BottledXavierLinear(in_features=self.hidden_size, out_features=argument_size).to(
        #     device=device)

        self.device = device
        self.emb = 0  # 一个batch 的嵌入表达初始化
        self.mask = 0

        # self.criterion = nn.CrossEntropyLoss(ignore_index=0)

        ## CRF - trigger
        kwargs = dict({'target_size': trigger_size, 'device': device})
        self.tri_CRF1 = CRF(**kwargs)
        self.tri_CRF2 = CRF(**kwargs)

        ## CRF - arugument
        kwargs_a = dict({'target_size': argument_size, 'device': device})
        self.arg_CRF1 = CRF(**kwargs_a)
        self.arg_CRF2 = CRF(**kwargs_a)
        # Ngramcnn
        # self.NgramCNN = NgramCNN(hidden_size=self.hidden_size)
    def predict_triggers(self, tokens_x_2d, entities_x_3d, postags_x_2d, head_indexes_2d, triggers_y_2d, arguments_2d,
                         adjm):

        # def get_Ngram_emb(self,emb,N):
        #
        #     batch_size, SEN_LEN, hidden_size = emb.size()
        #     hidden_size = hidden_size*2
        #     x = torch.zeros([batch_size, SEN_LEN, hidden_size],dtype = emb.dtype)
        #     # for i in range(batch_size):
        #     #     for j in range(SEN_LEN):
        #     #         x[i,j]=emb[i,max(j-N,0):min(j+N,SEQ_LEN-1)].mean(dim=0)
        #
        #     for j in range(SEN_LEN):
        #         cnnfeature=self.NgramCNN.forward(emb[:, max(j - N, 0):min(j + N, SEQ_LEN - 1),:])# [batch_size,hidden_size]
        #         Nmax, _ = emb[:,max(j-N,0):min(j+N,SEQ_LEN-1),:].max(dim=1)# [batch_size,hidden_size]
        #         x[:,j,:] = torch.cat([cnnfeature,Nmax],dim=-1)
        #
        #     return x.to(self.device)


        ## 字符ID [batch_size, seq_length]
        tokens_x_2d = torch.LongTensor(tokens_x_2d).to(self.device)
        ## 触发词标签ID [batch_size, seq_length]
        triggers_y_2d = torch.LongTensor(triggers_y_2d).to(self.device)
        ## [batch_size, seq_length]
        xlen = [max(x) for x in head_indexes_2d]
        head_indexes_2d = torch.LongTensor(head_indexes_2d).to(self.device)

        if self.training:
            self.PreModel.train()
            x_1, _ = self.PreModel(tokens_x_2d)
        else:
            self.PreModel.eval()
            with torch.no_grad():
                x_1, _ = self.PreModel(tokens_x_2d)



        batch_size = tokens_x_2d.shape[0]
        SEQ_LEN = x_1.size()[1]
        # [CLS]字符
        # sen_emb = torch.unsqueeze(x_1[:,0,:],dim=1).repeat(1, SEQ_LEN, 1)  # [batch,1,hidden_size]

        # 复数形式拆解
        x = torch.zeros(x_1.size(), dtype=x_1.dtype).to(self.device)
        for i in range(batch_size):
            ## 切片, 会改变位置 同时去除了[CLS]
            x[i] = torch.index_select(x_1[i], 0, head_indexes_2d[i])
            # x_2[i] = torch.index_select(x_2[i], 0, head_indexes_2d[i])


        mask = numpy.zeros(shape=[batch_size, SEQ_LEN], dtype=numpy.uint8)
        for i in range(len(xlen)):
            mask[i,:xlen[i]] = 1
        mask = torch.ByteTensor(mask).to(self.device)

        self.mask = mask
        ## [batch_size, SEQ_LEN, hidden_size*2]
        # n_gram_emb = get_Ngram_emb(self,x,5)

        ## emb = torch.cat([x,sen_emb,n_gram_emb],dim=-1) #hidden_size*3
        #emb = torch.cat([x, n_gram_emb], dim=-1)  # hidden_size*3

        emb = x # [batch_size, seq_len, hidden_size]
        trigger_logits1 = self.tri_fc1(emb)  # x: [batch_size, seq_len, trigger_size + 2 ]

        ## tri_CRF1 ##
        trigger_loss1 = self.tri_CRF1.neg_log_likelihood_loss(feats=trigger_logits1, mask=mask, tags=triggers_y_2d)
        _, trigger_hat_2d_crf1 = self.tri_CRF1.forward(feats=trigger_logits1, mask=mask)



        ## 根据触发词结果进行嵌入
        trigger_crf1_emb = self.trigger_embed(trigger_hat_2d_crf1)


        ## 将触发词嵌入加入BERT表达t         ## 将触发词嵌入加入BERT表达t
        ## [batch_size, SEQ_LEN , size+50]
        emb = torch.cat([emb, trigger_crf1_emb],dim=-1) # hidden_size + trigger_embsize
        trigger_logits2 = self.tri_fc2(emb)  # x: [batch_size, seq_len, trigger_size + 2 ]

        ## 滑动平均
        trigger_logits2 = trigger_logits2 * 0.8 + trigger_logits1 * 0.2

        ## tri_CRF2 ##
        trigger_loss2 = self.tri_CRF2.neg_log_likelihood_loss(feats=trigger_logits2, mask=mask, tags=triggers_y_2d)

        _, trigger_hat_2d_crf2 = self.tri_CRF2.forward(feats=trigger_logits2, mask=mask)

        trigger_crf2_emb = self.trigger_embed(trigger_hat_2d_crf2)

        trigger_hat_2d =  trigger_hat_2d_crf2

        self.emb = torch.cat([emb, trigger_crf2_emb],dim=-1) # hidden_size + trigger_embed*2

        trigger_loss = 0.5*(trigger_loss1 + trigger_loss2)

        argument_keys = {}
        sen_mask_id = []
        for i in range(batch_size):
            ## 列表 元素格式：[触发词开始位置，触发词结束位置，事件类型（34个）
            predicted_triggers = find_triggers([idx2trigger[trigger] for trigger in trigger_hat_2d[i].tolist()])
            for predicted_trigger in predicted_triggers:
                ## 预测-触发词开始位置，预测-触发词结束位置，预测-事件类型（文本）
                t_start, t_end, t_type_str = predicted_trigger
                ## 当预测的触发词 是正确的
                if (t_start, t_end, t_type_str) in arguments_2d[i]['events']:
                    for (a_start, a_end, a_type_idx) in arguments_2d[i]['events'][(t_start, t_end, t_type_str)]:
                        if (i, t_start, t_end, t_type_str) in argument_keys:
                            argument_keys [(i, t_start, t_end, t_type_str)].append((a_start, a_end, a_type_idx))
                        else:
                            argument_keys[(i, t_start, t_end, t_type_str)]= []
                            argument_keys [(i, t_start, t_end, t_type_str)].append((a_start, a_end, a_type_idx))
        return trigger_loss, triggers_y_2d, trigger_hat_2d, sen_mask_id, argument_keys



    def predict_arguments(self, sen_mask_id, argument_keys, arguments_2d, adjm):

        def get_sentence_positional_feature(self, argument_keys, seq_len):
            '''
            :param argument_keys:
            :return:[dim0, Seq_len, 768]
            '''
            max_len = 50
            dim0 = len(argument_keys)
            positions = numpy.zeros([dim0, seq_len], dtype=numpy.int64)
            for k, trigger in enumerate(argument_keys):
                (i, t_start, t_end, t_type_str) = trigger

                positions[k, :t_start] = range(max_len - t_start, max_len)
                positions[k, t_end:] = range(max_len, max_len + seq_len - t_end)

            positions = torch.LongTensor(positions).to(self.device)
            positions[positions<0] = 99
            positions[positions>99] = 99
            y = self.position_embed(positions)
            return y

        ## [batch_size, SEQ_LEN , hidden_size + trigger_embed*2]
        x = self.emb
        SEQ_LEN = x.size()[1]

        argument_hidden = []
        mask = []
        ## dim0 = 一个batch中 触发词的总数
        dim0 = len(argument_keys)
        arguments_y_2d= numpy.zeros([dim0,SEQ_LEN],dtype=numpy.int64)

        for k,trigger in enumerate(argument_keys):
            ( i, t_start, t_end, t_type_str) = trigger# 0 <= i < batch_size
            trigger_BERTemb = x[i, t_start:t_end, :self.hidden_size].mean(dim=0)#[hidden_size]
            trigger_BERTemb = torch.unsqueeze(trigger_BERTemb,dim=0).repeat(SEQ_LEN, 1)  # [SEQ_LEN,hidden_size]
            argument_hidden.append(torch.cat([x[i],trigger_BERTemb],dim=-1))    # [SEQ_LEN,hidden_size*2+tri_embsize*2]
            mask.append(self.mask[i])
            arguments_y_2d[k,:sum(self.mask[i].cpu())] = 1
            for (a_start, a_end, a_type_idx) in argument_keys[trigger]:
                arguments_y_2d[k, a_start: a_end] = a_type_idx

        argument_hidden = torch.stack(argument_hidden)


        ## 给argument_hidden加入触发词的位置嵌入
        ## [dim0,seq_len,pos_embsize]
        pos_emb= get_sentence_positional_feature(self, argument_keys, SEQ_LEN)
        ##  [dim0,seq_len, , hidden_size*2 + trigger_embed*2 + pos_embsize]
        argument_hidden =torch.cat([argument_hidden, pos_emb],dim=-1)


        ######### argument CRF1 #########
        argument_logits1 = self.arg_fc1(argument_hidden)
        arguments_y_2d = torch.LongTensor(arguments_y_2d).to(self.device)
        mask = torch.stack(mask).to(self.device)
        argument_loss1 = self.arg_CRF1.neg_log_likelihood_loss(feats=argument_logits1, mask= mask, tags=arguments_y_2d)
        _, argument_hat_1d_crf1 = self.arg_CRF1.forward(feats=argument_logits1, mask=mask)
        ###############################

        ## 根据第一次结果进行嵌入
        argument_emb  = self.trigger_embed(argument_hat_1d_crf1)
        ##  [dim0,seq_len, , hidden_size*2 + trigger_embed*2 + pos_embsize + argument_embsize]
        argument_hidden = torch.cat([argument_hidden, argument_emb], dim=-1)

        ######### argument CRF2（最终预测）#########
        argument_logits2 = self.arg_fc2(argument_hidden)

        ## 滑动平均
        argument_logits2 = argument_logits2 * 0.8 + argument_logits1 * 0.2

        argument_loss2 = self.arg_CRF2.neg_log_likelihood_loss(feats=argument_logits2, mask= mask, tags=arguments_y_2d)
        _, argument_hat_1d = self.arg_CRF2.forward(feats=argument_logits2, mask=mask)

        argument_loss=0.5*(argument_loss1 + argument_loss2)

        batch_size = len(arguments_2d)
        argument_hat_2d = [{'events': {}} for _ in range(batch_size)]
        for k,trigger in enumerate(argument_keys):
            (i, st, ed, event_type_str) = trigger
            if (st, ed, event_type_str) not in argument_hat_2d[i]['events']:
                argument_hat_2d[i]['events'][(st, ed, event_type_str)] = []
            result= find_argument(argument_hat_1d[k].cpu())
            for (e_st, e_ed, entity_type) in result:
                argument_hat_2d[i]['events'][(st, ed, event_type_str)].append((e_st, e_ed, entity_type))
        return argument_loss, arguments_y_2d, argument_hat_1d, argument_hat_2d


# Reused from https://github.com/lx865712528/EMNLP2018-JMEE
class MultiLabelEmbeddingLayer(nn.Module):
    def __init__(self,
                 num_embeddings=None, embedding_dim=None,
                 dropout=0.5, padding_idx=0,
                 max_norm=None, norm_type=2,
                 device=torch.device("cpu")):
        super(MultiLabelEmbeddingLayer, self).__init__()

        self.matrix = nn.Embedding(num_embeddings=num_embeddings,
                                   embedding_dim=embedding_dim,
                                   padding_idx=padding_idx,
                                   max_norm=max_norm,
                                   norm_type=norm_type)
        self.dropout = dropout
        self.device = device
        self.to(device)

    def forward(self, x):
        batch_size = len(x)
        seq_len = len(x[0])
        x = [self.matrix(torch.LongTensor(x[i][j]).to(self.device)).sum(0)
             for i in range(batch_size)
             for j in range(seq_len)]
        x = torch.stack(x).view(batch_size, seq_len, -1)

        if self.dropout is not None:
            return F.dropout(x, p=self.dropout, training=self.training)
        else:
            return x
