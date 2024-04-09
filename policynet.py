import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.functional as F
import torch.nn as nn
from torch_geometric.nn.conv import HypergraphConv

class HypergraphPointerNet(nn.Module):
    def __init__(self, args, statedim):
        super(HypergraphPointerNet, self).__init__()
        self.args = args
        self.hgnn = nn.ModuleList()
        for i in range(len(args.pnhid)):
            if (i==0):
                self.hgnn.append(HypergraphConv(statedim, args.pnhid[i]))
            else:
                self.hgnn.append(HypergraphConv(self.args.pnhid[i-1], self.args.pnhid[i]))
        self.weight1 = nn.Parameter(torch.randn(self.args.pnhid[i], self.args.pnhid[i]))
        self.weight2 = nn.Parameter(torch.randn(self.args.pnhid[i], self.args.pnhid[i]))
        self.weight3 = nn.Parameter(torch.randn(self.args.pnhid[i], 1))
        self.linear = nn.Linear(self.args.pnhid[i], 1)

    def forward(self, state, hyperedge_index, weight_matrix):
        # state.shape: (batch, node, statedim)
        x = state
        x_batch_split = []
        for i in range(self.args.batchsize):
            h = x[i]
            for layer in self.hgnn:
                h = torch.relu(layer(h, hyperedge_index, weight_matrix.long()))
            # x_batch_split: list, len: batchsize, x_batch_split[i].shape: (node, hidden_dim)
            x_batch_split.append(h)
        # x.shape: (batch, node, hidden)
        x = torch.stack(x_batch_split, dim=0)
        if self.args.use_pointer:
            # graph_emb: (batch, 1, hidden)
            graph_emb = x.sum(dim=1, keepdim=True)
            # logtis.shape: (batch, node)
            logits = torch.matmul(torch.tanh(torch.matmul(x, self.weight1) + torch.matmul(graph_emb, self.weight2)), self.weight3).squeeze(-1)
            return logits
        else:
            logits = self.linear(x)
            return logits