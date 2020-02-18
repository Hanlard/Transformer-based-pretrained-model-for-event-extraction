# @Author : bamtercelboo
# @Datetime : 2018/9/14 8:43
# @File : Sequence_Label.py
# @Last Modify Time : 2018/9/14 8:43
# @Contact : bamtercelboo@{gmail.com, 163.com}

"""
    FILE :  Sequence_Label.py
    FUNCTION : None
"""

import torch
import torch.nn as nn
import random
import numpy as np
import time
from models.BiLSTM import BiLSTM
from models.BiLSTM_CNN import BiLSTM_CNN
from models.CRF import CRF
from DataUtils.Common import *
torch.manual_seed(seed_num)
random.seed(seed_num)


class Sequence_Label(nn.Module):
    """
        Sequence_Label
    """

    def __init__(self, config):
        super(Sequence_Label, self).__init__()
        self.config = config
        # embed
        self.embed_num = config.embed_num
        self.embed_dim = config.embed_dim
        self.label_num = config.class_num
        self.paddingId = config.paddingId
        # dropout
        self.dropout_emb = config.dropout_emb
        self.dropout = config.dropout
        # lstm
        self.lstm_hiddens = config.lstm_hiddens
        self.lstm_layers = config.lstm_layers
        # pretrain
        self.pretrained_embed = config.pretrained_embed
        self.pretrained_weight = config.pretrained_weight
        # char
        self.use_char = config.use_char
        self.char_embed_num = config.char_embed_num
        self.char_paddingId = config.char_paddingId
        self.char_dim = config.char_dim
        self.conv_filter_sizes = self._conv_filter(config.conv_filter_sizes)
        self.conv_filter_nums = self._conv_filter(config.conv_filter_nums)
        assert len(self.conv_filter_sizes) == len(self.conv_filter_nums)
        # print(self.conv_filter_nums)
        # print(self.conv_filter_sizes)
        # exit()
        # use crf
        self.use_crf = config.use_crf

        # cuda or cpu
        self.device = config.device

        self.target_size = self.label_num if self.use_crf is False else self.label_num + 2

        if self.use_char is True:
            self.encoder_model = BiLSTM_CNN(embed_num=self.embed_num, embed_dim=self.embed_dim, label_num=self.target_size,
                                            paddingId=self.paddingId, dropout_emb=self.dropout_emb, dropout=self.dropout,
                                            lstm_hiddens=self.lstm_hiddens, lstm_layers=self.lstm_layers,
                                            pretrained_embed=self.pretrained_embed, pretrained_weight=self.pretrained_weight,
                                            char_embed_num=self.char_embed_num, char_dim=self.char_dim,
                                            char_paddingId=self.char_paddingId, conv_filter_sizes=self.conv_filter_sizes,
                                            conv_filter_nums=self.conv_filter_nums, device=self.device)
        else:
            self.encoder_model = BiLSTM(embed_num=self.embed_num, embed_dim=self.embed_dim, label_num=self.target_size,
                                        paddingId=self.paddingId, dropout_emb=self.dropout_emb, dropout=self.dropout,
                                        lstm_hiddens=self.lstm_hiddens, lstm_layers=self.lstm_layers,
                                        pretrained_embed=self.pretrained_embed, pretrained_weight=self.pretrained_weight,
                                        device=self.device)
        if self.use_crf is True:
            args_crf = dict({'target_size': self.label_num, 'device': self.device})
            self.crf_layer = CRF(**args_crf)

    @staticmethod
    def _conv_filter(str_list):
        """
        :param str_list:
        :return:
        """
        int_list = []
        str_list = str_list.split(",")
        for str in str_list:
            int_list.append(int(str))
        return int_list

    def forward(self, word, char, sentence_length, train=False):
        """
        :param char:
        :param word:
        :param sentence_length:
        :param train:
        :return:
        """
        if self.use_char is True:
            encoder_output = self.encoder_model(word, char, sentence_length)
            return encoder_output
        else:
            encoder_output = self.encoder_model(word, sentence_length)
            return encoder_output


