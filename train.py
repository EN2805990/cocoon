import os
import sys


import numpy as np
import torch
import argparse
from pprint import pformat
from torch.distributions import Categorical
import torch.nn.functional as F
from src.dataloader import GraphLoader
from src.player import Player
from src.env import Env
from src.policynet import *
from src.rewardshaper import RewardShaper
from src.common import *
from src.utils import *

switcher = {'hpn': HypergraphPointerNet}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nhid",type=int,default=64)
    parser.add_argument("--pnhid",type=str,default='256+256+256')
    parser.add_argument("--dropout",type=float,default=0.2)
    parser.add_argument("--pdropout",type=float,default=0.0)
    parser.add_argument("--lr",type=float,default=3e-2)
    parser.add_argument("--rllr",type=float,default=1e-2)
    parser.add_argument("--entcoef",type=float,default=0)
    parser.add_argument("--frweight",type=float,default=1e-3)
    parser.add_argument("--batchsize",type=int,default=4)
    parser.add_argument("--budgets",type=str,default="20",help="budget per class")
    parser.add_argument("--ntest",type=int,default=1000)
    parser.add_argument("--nval",type=int,default=500)
    parser.add_argument("--datasets",type=str,default="ba3000")
    parser.add_argument("--metric",type=str,default="microf1")
    parser.add_argument("--remain_epoch",type=int,default=35,help="continues training $remain_epoch"
                                                                  " epochs after all the selection")
    parser.add_argument("--shaping",type=str,default="234",help="reward shaping method, 0 for no shaping;"
                                                              "1 for add future reward,i.e. R= r+R*gamma;"
                                                              "2 for use finalreward;"
                                                              "3 for subtract baseline(value of curent state)"
                                                              "1234 means all the method is used,")
    parser.add_argument("--logfreq",type=int,default=10)
    parser.add_argument("--maxepisode",type=int,default=20000)
    parser.add_argument("--save",type=int,default=0)
    parser.add_argument("--savename",type=str,default="tmp")
    parser.add_argument("--policynet",type=str,default='hpn')
    parser.add_argument("--multigraphindex", type=int, default=0)

    parser.add_argument("--use_entropy",type=int,default=1)
    parser.add_argument("--use_degree",type=int,default=0)
    parser.add_argument("--use_local_diversity",type=int,default=1)
    parser.add_argument("--use_select",type=int,default=1)
    parser.add_argument("--use_incidence_matrix",type=int,default=0)
    parser.add_argument("--use_budget", type=int, default=1)
    parser.add_argument("--use_pointer",type=int,default=1)

    parser.add_argument('--pg', type=str)
    parser.add_argument('--ppo_epoch', type=int, default=5)

    parser.add_argument('--gpu', type=int, default=5)
    parser.add_argument('--schedule', type=int, default=0)

    args = parser.parse_args()
    # logargs(args,tablename="config")
    args.pnhid = [int(n) for n in args.pnhid.split('+')]
    
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    return args

class SingleTrain(object):
    def __init__(self, args):
        self.args = args
        self.datasets = self.args.datasets.split("+")
        self.budgets = [int(x) for x in self.args.budgets.split("+")]
        self.graphs, self.players, self.rshapers, self.accmeters = [], [], [], []
        for i, dataset in enumerate(self.datasets):
            g = GraphLoader(dataset)
            g.process()
            self.graphs.append(g)
            p = Player(g, args).cuda()
            self.players.append(p)
            self.rshapers.append(RewardShaper(args))
            self.accmeters.append(AverageMeter("accmeter", ave_step=100))
        self.env = Env(self.players, args)

        self.policy = switcher[args.policynet](self.args, self.env.statedim).cuda()

        self.opt = torch.optim.Adam(self.policy.parameters(), lr=self.args.rllr)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.opt, [1000, 3000], gamma=0.1, last_epoch=-1)
        self.accmeter = (AverageMeter("aveaccmeter", ave_step=100))

    def jointtrain(self, maxepisode):

        for episode in range(1, maxepisode):
            for playerid in range(len(self.datasets)):
                # logp_actions.shape: (budget, batch)
                shaperewards, logp_actions, p_actions = self.playOneEpisode(episode, playerid=playerid) #TODO shaperewards
                if episode > 10:
                    pass
                    # loss = self.finishEpisode(shaperewards, logp_actions, p_actions)#p_actions在ppo中用
                else:
                    loss = None
                if self.args.save==1 and self.accmeter.should_save():  
                    logger.warning("saving!")
                    torch.save(self.policy.state_dict(), "models/{}.pkl".format(self.args.policynet+self.args.savename)) 
            if (episode % 100 == 1):
                torch.save(self.policy.state_dict(), "models/{}.pkl".format(self.args.policynet+self.args.savename+'_'+str(episode)))

    def playOneEpisode(self, episode, playerid=0):

        self.playerid = playerid
        self.env.reset(playerid)
        rewards, logp_actions, p_actions = [], [], []
        self.states, self.actions, self.pools = [], [], []
        # action_index: (batch, budget)
        self.action_index = np.zeros([self.args.batchsize, self.budgets[playerid]])
        for epoch in range(self.budgets[playerid]):
            # state.shape: (batch, node, 5)
            state = self.env.getState(playerid)
            self.states.append(state)
            pool = self.env.players[playerid].getPool()
            self.pools.append(pool)

            # logits.shape: (batch, node) (10, 2708)
            logits = self.policy(state, self.graphs[playerid].hyperedge_index, self.graphs[playerid].weight_matrix)
            # pool: list, len(pool): 2, pool[0].shape: (12080)
            # action, logp_action, p_action.shape: (batch) (10)
            action, logp_action, p_action = self.selectActions(logits, pool)
            self.action_index = action.detach().cpu().numpy()
            logp_actions.append(logp_action)
            p_actions.append(p_action)
            rewards.append(self.env.step(action, playerid) - sum(rewards) if epoch>0 else self.env.step(action, playerid))

        # logp_actions: (budget, batch)
        logp_actions = torch.stack(logp_actions)
        # p_actions: (budget, batch)
        p_actions = torch.stack(p_actions)
        # finalrewards.shape: (batch)
        finalrewards = self.env.players[playerid].validation()
        micfinal = mean_std(finalrewards)
        self.accmeters[playerid].update(micfinal)
        self.accmeter.update(micfinal)
        if episode % self.args.logfreq == 0:
            logger.info("episode {}, playerid {}. num hyperedge covered: {}, ave num hyperedge: {}".format(episode, playerid, micfinal, self.accmeters[playerid]()))
        shapedrewards = self.rshapers[playerid].reshape(rewards)
        return shapedrewards, logp_actions, p_actions
    
    def finishEpisode(self, rewards, logp_actions, p_actions):
        # rewards: (budget, batch)
        # rewards = torch.from_numpy(rewards).cuda().type(torch.float32)
        if (self.args.pg == 'reinforce'):
            # logp_actions.shape: (budget, batch)
            losses = logp_actions * rewards
            loss = -torch.mean(torch.sum(losses, dim=0))
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            if self.args.schedule:
                self.scheduler.step()
        elif (self.args.pg == 'ppo'):
            raise ValueError("No PPO!")

        
    def selectActions(self, logits, pool):
        # logits.shape: (batch, node)
        # pool: list, len(pool): 2, pool[0].shape: (12080)
        # valid_logits.shape: (10, 1208) -> (10, 1207) -> (10, 1206)
        valid_logits = logits[pool].reshape(self.args.batchsize, -1)
        max_logits = torch.max(valid_logits, dim=1, keepdim=True)[0].detach()
        valid_logits = valid_logits - max_logits
        valid_probs = torch.nn.functional.softmax(valid_logits, dim=1)
        self.valid_probs = valid_probs
        # pool.shape: (10, 1208) -> (10, 1207) -> (10, 1206)
        pool = pool[1].reshape(self.args.batchsize, -1)
        assert pool.size() == valid_probs.size()

        # valid_probs.shape: (10, 1208) -> (10, 1207) -> (10, 1206)
        m = Categorical(valid_probs)
        # action_inpool.shape: (batch)
        action_inpool = m.sample()
        self.actions.append(action_inpool)
        # logprob.shape: (batch)
        logprob = m.log_prob(action_inpool)
        # prob.shape: (batch)
        prob = valid_probs[list(range(self.args.batchsize)), action_inpool]
        action = pool[[x for x in range(self.args.batchsize)], action_inpool]
        # action, logprob, prob.shape: (batch)
        return action, logprob, prob


if __name__ == "__main__":

    args = parse_args()
    singletrain = SingleTrain(args)
    singletrain.jointtrain(args.maxepisode)