# This file contains the implementation of FedProx algorithm
# Fedrated optimization in heterogeneous networks(https://arxiv.org/abs/1812.06127)
# FedProx: proximal term is added to punish updates that are far away from global model
import copy
import time
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm_
from algorithms import ClientBase, ServerBase   
from utils import AverageMeter
import torch

class FedProx(ServerBase):
    def __init__(self, args) -> None:
        super(FedProx, self).__init__(args, Client)


class Client(ClientBase):
    def __init__(self, args, id, trainset) -> None:
        super(Client, self).__init__(args, id, trainset)
        self.mu = args.mu
        
    def train(self, round_idx, lr, state_dict):
        self.prepare(lr, state_dict)
        
        # Store the global model parameters for proximal term calculation
        global_params = copy.deepcopy(state_dict)
        
        if(round_idx%5==0 and round_idx!=0 and self.local_steps>=10):
            self.local_steps-=5
            
        self.model.train(True)
        loss_meter = AverageMeter()
        
        for step in range(self.local_steps):
            loss_meter.reset()
            for data in self.loader:
                x, y = data['x'].to(self.device), data['y'].to(self.device)
                if len(y) < 2:
                    continue
                    
                logits = self.model(x)
                
                # Calculate the main task loss
                task_loss = self.ls_func(logits, y)
                
                # Calculate proximal term
                proximal_term = 0
                for name, w in self.model.named_parameters():
                    w_t = global_params[name].to(self.device)
                    proximal_term += torch.sum(torch.pow(w - w_t, 2))
                    
                # Total loss = task_loss + (mu/2) * proximal_term
                loss = task_loss + (self.mu / 2) * proximal_term
                
                self.optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(self.model.parameters(), self.clip_grad)
                self.optimizer.step()
                
                loss_meter.update(loss.item(), y.size(0))
                
            self.printer.info(f'C{self.id:<2d}:{step}/{self.local_steps} >> avg loss {loss_meter.avg:.2f}')
            
        self.model.to('cpu')


    
        
