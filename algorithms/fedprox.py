# This file contains the implementation of FedProx algorithm
# Fedrated optimization in heterogeneous networks(https://arxiv.org/abs/1812.06127)
# FedProx: proximal term is added to punish updates that are far away from global model
import copy
import time
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm_
from algorithms import ClientBase, ServerBase   
from utils import AverageMeter

class FedProx(ServerBase):
    def __init__(self, args) -> None:
        super(FedProx, self).__init__(args, Client)


class Client(ClientBase):
    def __init__(self, args, id, trainset) -> None:
        super(Client, self).__init__(args, id, trainset)
        self.mu = args.mu


    
        
