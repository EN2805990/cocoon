import os
import pandas as pd
import numpy as np
import torch


class GraphLoader(object):

    def __init__(self, name, root="./data", undirected=False):  #TODO root
        self.name = name
        self.undirected = undirected
        self.dirname = os.path.join(root, name)
        self.prefix = os.path.join(root, name, name)
        self._load()

    def _loadConfig(self):
        file_name = os.path.join(self.dirname, f"{self.name}config.txt")
        self.stat = dict()
        with open(file_name, 'r') as f:
            for l in f:
                if len(l[0])!=0:
                    key, value = l.strip().split(' ')
                    self.stat[key] = int(value)

    def _loadGraph(self):
        # 读取文件
        pd_hyperedgeList = pd.read_csv(self.prefix+".hyperedgelist", sep='\t', header=None, names=['node', 'hyperedge'])
        self.hyperedge_index = pd_hyperedgeList.to_numpy().transpose(1,0)
        # 初始化关联矩阵
        self.incidence_matrix = np.zeros((self.stat['num_node'], self.stat['num_hyperedge']))
        # 填充关联矩阵
        for _, row in pd_hyperedgeList.iterrows():
            self.incidence_matrix[row['node'], row['hyperedge']] = 1

        pd_hyperedgeweight = pd.read_csv(self.prefix+".hyperedgeweight", sep='\t', header=None, names=['hyperedge', 'weight'])
        self.weight_matrix = np.zeros((self.stat['num_hyperedge']))
        for _, row in pd_hyperedgeweight.iterrows():
            self.weight_matrix[row['hyperedge']] = row['weight']

    def _toTensor(self, device=None):
        if device is None:
            self.incidence_matrix = torch.from_numpy(self.incidence_matrix).float().cuda()
            self.weight_matrix = torch.from_numpy(self.weight_matrix).float().cuda()
            self.hyperedge_index = torch.from_numpy(self.hyperedge_index).long().cuda()

    def _load(self):
        self._loadConfig()
        self._loadGraph()

    def process(self):
        self._toTensor()
