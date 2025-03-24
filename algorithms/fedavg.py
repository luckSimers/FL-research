# This file contains the implementation of FedAvg algorithm
import copy
import time
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm_
from algorithms.base import ClientBase, ServerBase
from algorithms.baseplus import ClientBasePlus, ServerBasePlus
from utils import *



class FedAvg(ServerBase):
    def __init__(self, args) -> None:
        super(FedAvg, self).__init__(args, Client)
    

class Client(ClientBase):
    def __init__(self, args, id, trainset) -> None:
        super(Client, self).__init__(args, id, trainset)

    def train(self, round_idx, lr, state_dict):
            self.prepare(lr, state_dict)
            self.model.train(True)
            loss_meter = AverageMeter()
            local_steps=self.local_steps
            if(round_idx==0):
                local_steps=self.local_steps+50
            for step in range(local_steps):
                loss_meter.reset()
                for data in self.loader:
                    x, y = data['x'].to(self.device), data['y'].to(self.device)
                    if len(y) < 2:
                        continue
                    logits = self.model(x)
                    loss = self.ls_func(logits, y)
                    self.optimizer.zero_grad()
                    loss.backward()
                    clip_grad_norm_(self.model.parameters(), self.clip_grad)
                    self.optimizer.step()
                    loss_meter.update(loss.item(), y.size(0))
                self.printer.info(f'C{self.id:<2d}:{step}/{self.local_steps} >> avg loss {loss_meter.avg:.2f}')
            self.model.to('cpu')
    
