import torch
import torch.nn as nn
import time
import torch.nn.functional as F
import numpy as np

class Player(nn.Module):
    def __init__(self, G, args):
        super(Player, self).__init__()
        self.G = G
        self.args = args
        self.batchsize = args.batchsize

        self.reset()
        self.count = 0

    def query(self, nodes):
        self.count += 1
        # nodes.shape: (batch)
        self.trainmask[[x for x in range(self.batchsize)], nodes] = 1.


    def getPool(self):
        mask = self.trainmask
        row, col = torch.where(mask<0.1)
        return row, col
    

    def validation(self):
        """
        输出trainmask覆盖的超边数
        """
        # trainmask.shape: (batch, nodes)
        assert self.count==self.trainmask[0].sum()
        # 结果形状: [batchsize, num_hyperedge]
        covered_hyperedges = torch.matmul(self.trainmask, self.G.incidence_matrix)

        # 将结果转换为布尔值，以确定是否有覆盖
        # covered_hyperedges_bool = covered_hyperedges > 0
        covered_hyperedges_bool = torch.where(covered_hyperedges>0.5, 1, 0).float()
        covered_hyperedges_count = torch.matmul(covered_hyperedges_bool, self.G.weight_matrix.unsqueeze(-1)).sum(dim=1)
        return covered_hyperedges_count

    def reset(self):
        self.trainmask = torch.zeros((self.batchsize, self.G.stat['num_node'])).to(torch.float).cuda()
        self.count = 0
