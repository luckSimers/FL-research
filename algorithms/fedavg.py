# This file contains the implementation of FedAvg algorithm
import copy
import time
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm_
from algorithms import ClientBase, ServerBase
from utils import *



class FedAvg(ServerBase):
    def __init__(self, args) -> None:
        super(FedAvg, self).__init__(args, Client)
    

class Client(ClientBase):
    def __init__(self, args, id, trainset) -> None:
        super(Client, self).__init__(args, id, trainset)
        self.clip_grad = args.clip_grad
        self.loss = nn.CrossEntropyLoss()

    def train(self, round_idx, lr, state_dict):
        self.prepare(lr, state_dict)
        
        self.model.train(True)
        loss_meter = AverageMeter()
        for step in range(self.local_steps):
            loss_meter.reset()
            for x, y in self.loader:
                x, y = x.to(self.device), y.to(self.device)
                if len(y) < 2:
                    continue
                logits = self.model(x)
                loss = self.loss(logits, y)
                self.optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(self.model.parameters(), self.clip_grad)
                self.optimizer.step()
                loss_meter.update(loss.item(), y.size(0))
            self.printer.info(f'C{self.id:<2d}:{step}/{self.local_steps} >> avg loss {loss_meter.avg:.2f}')
        self.model.to('cpu')
