# Muti-head Attention 机制的实现
from math import sqrt
import torch
import torch.nn as nn


class Self_Attention_Muti_Head(nn.Module):
    # input : batch_size * seq_len * input_dim
    # q : batch_size * input_dim * dim_k
    # k : batch_size * input_dim * dim_k
    # v : batch_size * input_dim * dim_v
    def __init__(self, input_dim, dim_k, dim_v, nums_head):
        super(Self_Attention_Muti_Head, self).__init__()
        assert dim_k % nums_head == 0
        assert dim_v % nums_head == 0
        self.q = nn.Linear(input_dim, dim_k)
        self.k = nn.Linear(input_dim, dim_k)
        self.v = nn.Linear(input_dim, dim_v)

        self.nums_head = nums_head
        self.dim_k = dim_k
        self.dim_v = dim_v
        self._norm_fact = 1 / sqrt(dim_k)

    def forward(self, x):
        # reshape(-1, m, n)中-1表示自动计算reshape后张量该纬度的数值，下例-1即为自动计算head数，
        # 总数为batch_size * seq_len * dim_k or dim_v,
        # reshape后为nums_head * batch_size * seq_len * dim_k//nums_head or dim_v//nums_head
        Q = self.q(x).reshape(-1, x.shape[0], x.shape[1], self.dim_k // self.nums_head)
        K = self.k(x).reshape(-1, x.shape[0], x.shape[1], self.dim_k // self.nums_head)
        V = self.v(x).reshape(-1, x.shape[0], x.shape[1], self.dim_v // self.nums_head)
        print(x.shape)
        print(Q.size())

        atten = nn.Softmax(dim=-1)(torch.matmul(Q, K.permute(0, 1, 3, 2)))  # Q * K.T() # batch_size * seq_len * seq_len

        output = torch.matmul(atten, V).reshape(x.shape[0], x.shape[1],
                                                -1)  # Q * K.T() * V # batch_size * seq_len * dim_v

        return output