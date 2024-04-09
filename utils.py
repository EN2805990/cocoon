import torch


class AverageMeter(object):
    def __init__(self,name='',ave_step=10):
        self.name = name
        self.ave_step = ave_step
        self.history =[]
        self.history_extrem = None
        self.S=5

    def update(self,data):
        if data is not None:
            self.history.append(data)

    def __call__(self):
        if len(self.history) == 0:
            value =  None
        else:
            cal=self.history[-self.ave_step:]
            value = sum(cal)/float(len(cal))
        return value

    def should_save(self):
        if len(self.history)>self.S*2 and sum(self.history[-self.S:])/float(self.S)> sum(self.history[-self.S*2:])/float(self.S*2):
            if self.history_extrem is None :
                self.history_extrem =sum(self.history[-self.S:])/float(self.S)
                return False
            else:
                if self.history_extrem < sum(self.history[-self.S:])/float(self.S):
                    self.history_extrem = sum(self.history[-self.S:])/float(self.S)
                    return True
                else:
                    return False
        else:
            return False
        
def mean_std(x):
    mean_batch = torch.mean(x).item()
    return mean_batch