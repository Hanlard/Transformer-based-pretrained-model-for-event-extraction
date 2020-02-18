import torch
import torch.nn as nn
import numpy as np
from data_load import all_triggers, all_arguments

class dmcnn_t(nn.Module):
    def __init__(self):
        super(dmcnn_t, self).__init__()
        self.keep_prob = 0.5
        self.feature_t = 200            # num of trigger features in cnn
        self.feature_a = 300            # num of argument features in cnn
        self.pf_t = 5                   # dim of pf(位置嵌入维度) in trigger classification
        self.pf_a = 5                   # dim of pf(位置嵌入维度) in argument classification
        self.ef_a = 5                   # 事件类型嵌入维度
        self.window_t = 3               # 触发词窗长 window size in cnn
        self.window_a = 3               # 角色词窗长 window size in cnn
        self.hidden_size =  768
        self.trigger_size = len(all_triggers)
        self.argument_size = len(all_arguments)
        self.conv = nn.Conv1d(self.hidden_size, self.feature_t, self.window_t, bias=True, padding=1)
        self.L = nn.Linear(2*self.feature_t, self.trigger_size , bias=True)
        self.dropout = nn.Dropout(p=self.keep_prob)

    def pooling(self, conv):
        # [batch_size, seq_len, self.feature_t]
        conv_l = []
        seq_len = conv.size()[1]
        batch_size = conv.size()[0]
        for i in range(1, seq_len):
            x = conv.index_select(1, torch.tensor(list(range(i))).type(torch.LongTensor).to("cuda"))
            x, _ = x.max(dim=1)
            y = conv.index_select(1, torch.tensor(list(range(i, seq_len))).type(torch.LongTensor).to("cuda"))
            y, _ = y.max(dim=1)
            z = torch.cat([x, y], 0).t().contiguous().view([batch_size, 1, -1])
            conv_l.append(z)
        conv_l.append(torch.cat([conv.max(dim=1)[0], conv.max(dim=1)[0]], dim=1).view([batch_size, 1, -1]))
        pooled = torch.stack(conv_l)
        pooled = pooled.squeeze(dim=-2)
        pooled = pooled.permute(1, 0, 2).to("cuda")

        return pooled

    def forward(self,emb):
        x= emb
        x = torch.tanh(self.conv(x.permute(0, 2, 1)))       # [batch, feature, sen-2]
        x = x.permute(0, 2, 1).to("cuda")                            # [batch, sen-2, feature]
        x = self.pooling(x)                                 # [batch, 2*feature]]
        x = self.dropout(x)
        logits = self.L(x)                                       # [batch, trigger]
        return logits

########test#########

# import torch
# import torch.nn as nn
# import numpy as np
# from data_load import all_triggers, all_arguments
#
# a1=np.ones([3,6])
# a2=np.ones([4,6])*2
# conv = torch.cat([torch.tensor(a1),torch.tensor(a2)],dim=0)
# conv=conv.unsqueeze(dim=0)
# # conv=torch.cat([conv,conv],dim=0)
# conv.size()
#
#
# conv_l = []
# seq_len = conv.size()[1]
# batch_size = conv.size()[0]
#
# for i in range(1,seq_len):
#     x = conv.index_select(1, torch.tensor(list(range(i))).type(torch.LongTensor))
#     x, _ = x.max(dim=1)
#     y = conv.index_select(1, torch.tensor(list(range(i, seq_len))).type(torch.LongTensor))
#     y, _ = y.max(dim=1)
#     z = torch.cat([x, y], 0).t().contiguous().view([batch_size ,1, -1])
#     conv_l.append(z)
# conv_l.append(torch.cat([conv.max(dim=1)[0],conv.max(dim=1)[0]],dim=1).view([batch_size ,1, -1]))
# pooled = torch.stack(conv_l)
# pooled=pooled.squeeze(dim=-2)
# pooled = pooled.permute(1, 0, 2).to("cuda")
# pooled.size()