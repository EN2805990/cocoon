import torch

class Env(object):

    def __init__(self, players, args):
        self.players = players
        self.args = args
        self.nplayer = len(self.players)
        self.graphs = [p.G for p in self.players]
        self.budgets = [int(x) for x in args.budgets.split("+")]
        self.statedim = self.getState(0).size(-1)
    
    def step(self, actions, playerid=0):
        # actions.shape: (batch)
        p = self.players[playerid]
        p.query(actions)
        # cover_num_hyperedge.shape: (batch)
        num_hyperedge_covered = p.validation()
        return num_hyperedge_covered
    
    def getState(self, playerid=0):
        p = self.players[playerid]
        budget = self.budgets[playerid]
        state = self.makeState(p.trainmask, p.G, budget)
        return state
    
    def makeState(self, selected, G, budget):
        # selected is p.trainmask. shape: (batch, node)
        batchsize = selected.size(0)
        num_node = selected.size(1)
        incidence_matrix = G.incidence_matrix
        weight_matrix = G.weight_matrix
        num_hyperedge = incidence_matrix.size(1)
        row, col = torch.where(selected == 1)
        features = []
        if self.args.use_select:
            features.append(selected.unsqueeze(-1))
        if self.args.use_degree:
            # G.shape: (node, edge)
            # G_batch.shape: (batch, node, edge)
            incidence_matrix_batch = incidence_matrix.unsqueeze(0).repeat(batchsize, 1, 1)

            # (batch, node)*(node, num_hyperedge) -> (batch, num_node, num_hyperedge)
            hyperedge_covered_batch = torch.matmul(selected, incidence_matrix).unsqueeze(1).repeat(1, num_node, 1)
            incidence_matrix_batch_remain = torch.where(incidence_matrix_batch>hyperedge_covered_batch, incidence_matrix_batch, torch.zeros_like(incidence_matrix_batch))
            deg = torch.matmul(incidence_matrix_batch_remain, weight_matrix.unsqueeze(1))
            features.append(deg)


            # # 检查是否有被选中的元素
            # if row.nelement() == 0 or col.nelement() == 0:
            #     # 如果没有被选中的元素，处理方式取决于您的需求
            #     # deg = torch.zeros((batchsize, num_node, 1)).cuda()
            #     deg = incidence_matrix_batch.sum(dim=-1, keepdim=True)
            # else:
            #     # 有被选中的元素，继续正常处理
            #     # deg.shape: (batch, node, 1)
            #     deg = torch.where((incidence_matrix_batch - incidence_matrix_batch[row,col,:].reshape(batchsize, -1, num_hyperedge).sum(dim=1, keepdim=True))>0, incidence_matrix_batch, torch.zeros((batchsize,num_node,num_hyperedge)).cuda()).sum(dim=-1, keepdim=True)  #TODO
            #     # incidence_matrix_batch[row,col,:].reshape(batchsize, -1, num_hyperedge).sum(dim=1, keepdim=True)  #TODO
            # features.append(deg)
        if self.args.use_incidence_matrix:
            features.append(incidence_matrix.unsqueeze(0).repeat(batchsize, 1, 1))
        if self.args.use_budget:
            # (batch, node, 1)
            features.append(torch.full((batchsize, num_node, 1), budget).float().cuda() - selected.unsqueeze(-1).sum(0, keepdim=True))
        # state: (batch, node, state_dim)
        state = torch.cat(features, dim=-1)
        return state
    
    def reset(self, playerid=0):
        self.players[playerid].reset()