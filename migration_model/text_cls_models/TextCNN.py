# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from data_load import  all_arguments

class TextCNN(nn.Module):
    def __init__(self):
        super(TextCNN, self).__init__()
        self.dropout = 0.5                                              # 随机失活
        self.filter_sizes = (2, 3, 4)                                   # 卷积核尺寸
        self.num_filters = 256                                          # 卷积核数量(channels数)
        self.embed = 768
        self.num_classes = len(all_arguments) + 2

        self.convs = nn.ModuleList(
            [nn.Conv2d(1, self.num_filters, (k, self.embed)) for k in self.filter_sizes])
        # self.dropout = nn.Dropout(0.5)
        # self.fc = nn.Linear(self.num_filters * len(self.filter_sizes), self.num_classes)

    def conv_and_pool(self, x, conv):

        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)

        return x

    def forward(self, x):

        out = x.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        # out = self.dropout(out)
        # out = self.fc(out)

        return out

class NgramCNN(nn.Module):
    def __init__(self,hidden_size):
        super(NgramCNN, self).__init__()
        self.filter_sizes = (2, 3, 4, 5)                                   # 卷积核尺寸
        self.num_filters = 256                                          # 卷积核数量(channels数)
        self.embed = hidden_size
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, self.num_filters, (k, self.embed)) for k in self.filter_sizes])

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = x.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        return out
