import torch
from torchcrf import CRF
seq_length, batch_size, num_tags = 3, 2, 5
emissions = torch.autograd.Variable(torch.randn(seq_length, batch_size, num_tags), requires_grad=True)
tags = torch.autograd.Variable(torch.LongTensor([[0, 1], [2, 4], [3, 1]]))  # (seq_length, batch_size)
model = CRF(num_tags=5)