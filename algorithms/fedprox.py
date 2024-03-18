# This file contains the implementation of FedProx algorithm
# Fedrated optimization in heterogeneous networks(https://arxiv.org/abs/1812.06127)
# FedProx: proximal term is added to punish updates that are far away from global model
import copy
import time
import logging
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader
from utils import *
from algorithms import ClientBase, ServerBase


class FedProx(ServerBase):
    def __init__(self, args) -> None:
        super(FedProx, self).__init__(args, Client)
    
    @staticmethod
    def get_argument():
        return [
            Special_Argument('--mu', type=float, default=0.1, help='weight of proximal term'),

        ]

class Client(ClientBase):
    def __init__(self, args, id, trainset) -> None:
        super(Client, self).__init__(args, id, trainset)
        self.clip_grad = args.clip_grad
        self.loss = nn.CrossEntropyLoss()
        self.mu = args.mu


    def train(self, round_idx, lr, state_dict):
        st = time.time()
        self.prepare(lr, state_dict)
        loader = DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True)
        anchor = copy.deepcopy(self.model).to(self.device)
        self.model.train(True)
        for step in range(self.local_steps):
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                if len(y) < 2:
                    continue
                logits = self.model(x)
                loss = self.loss(logits, y)
                self.optimizer.zero_grad()
                loss.backward()
                for p, anchor_p in zip(self.model.parameters(), anchor.parameters()):
                    p.grad += self.mu * (p.data - anchor_p.data.detach())
                if self.clip_grad > 0:
                    clip_grad_norm_(self.model.parameters(), self.clip_grad)
                self.optimizer.step()
        self.model.to('cpu')
        logging.info(f'C{self.id:<2d}>> training cost {(time.time() - st)/60:.2f} min')
