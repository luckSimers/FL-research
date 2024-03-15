# This file contains the implementation of FedAvg algorithm
import copy
import time
import logging
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import *
from algorithms import ClientBase, ServerBase
from torch.nn.utils.clip_grad import clip_grad_norm_


class FedAvg(ServerBase):
    def __init__(self, args) -> None:
        super(FedAvg, self).__init__(args, Client)
    
    @staticmethod
    def get_argument():
        return [
            
        ]

class Client(ClientBase):
    def __init__(self, args, id, trainset) -> None:
        super(Client, self).__init__(args, id, trainset)
        self.clip_grad = args.clip_grad
        self.loss = nn.CrossEntropyLoss()

    def train(self, round_idx, lr, state_dict):
        st = time.time()
        self.prepare(lr, state_dict)
        loader = DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True)
        self.model.train(True)
        for step in range(self.local_steps):
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                loss = self.loss(logits, y)
                self.optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(self.model.parameters(), self.clip_grad)
                self.optimizer.step()
                
        logging.info(f'C{self.id:<2d}>> training cost {(time.time() - st)/60:.2f} min')
        self.model.to('cpu')
