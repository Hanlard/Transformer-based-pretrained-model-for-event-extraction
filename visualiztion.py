import torch

import torch.nn as nn

from tensorboardX import SummaryWriter

from torch.autograd import Variable


class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()

        self.fc3 = nn.Linear(84, 10)

    # 定义前向传播过程，输入为x

    def forward(self, x):

        x = self.fc3(x)

        return x


dummy_input = Variable(torch.rand(3, 224, 84))

model = LeNet()

with SummaryWriter(comment='resnet34') as w:
    w.add_graph(model, input_to_model=dummy_input)