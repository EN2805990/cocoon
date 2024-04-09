import torch


class RewardShaper(object):
    def __init__(self, args):
        self.args = args
    
    def reshape(self, rewards):
        # rewards: list, len(rewards): budget, rewards[0].shape:(batch)
        returns = []
        R = torch.zeros((self.args.batchsize)).cuda()
        for r in reversed(rewards):
            R = r + R
            returns.insert(0, R)
        reshape_rewards = torch.stack(returns, dim=0)
        # returns: (budget, batch)
        return reshape_rewards