# This file contains the implementation of FedAvg algorithm
import copy
import time
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm_
from algorithms.baseplus import ClientBasePlus, ServerBasePlus
from utils import *



class FedAvg(ServerBasePlus):
    def __init__(self, args) -> None:
        super(FedAvg, self).__init__(args, Client)
    

class Client(ClientBasePlus):
    def __init__(self, args, id, trainset) -> None:
        super(Client, self).__init__(args, id, trainset)

    
