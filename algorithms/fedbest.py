# T
import copy
import time
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm_
from algorithms import ClientBase, ServerBase   
from utils import AverageMeter
import os
import copy
import time
import torch
import numpy as np
import torch.nn as nn
import torch.cuda as cuda
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.clip_grad import clip_grad_norm_
from algorithms.base import ClientBase, ServerBase
from utils import get_optimizer, get_net_builder
from datasets import fetch_dataset, split_dataset, SubDataset
from utils import AverageMeter, make_batchnorm_stats
from nngeometry.metrics import FIM
from nngeometry.object import PMatDiag
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

def cosine_similarity(a, b):
    """ 计算两个向量的余弦相似度 """
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

def calculate_average_cosine_similarity(list1, list2):
    """ 计算两个列表中每对向量的余弦相似度并返回平均值 """
    similarities = []
    
    for vec1, vec2 in zip(list1, list2):
        sim = cosine_similarity(vec1, vec2)
        similarities.append(sim)
    
    # 计算平均值
    avg_similarity = sum(similarities) / len(similarities)
    return avg_similarity

class FedBest(ServerBase):
    def __init__(self, args) -> None:
        super(FedBest, self).__init__(args, Client)
        self.best_acc_fc={}
        
    def run(self):
        if self.sBN:
            super.make_batchnorm_stats(self.batchnorm_dataset, self.global_model, self.device)
        for round_idx in range(self.global_rounds):
            st_time = time.time()
            self.selected_clients = self.select_clients(round_idx)
            self.selected_clients.sort()
            lr = self.scheduler.get_last_lr()[0]
            model_dict = self.global_model.state_dict()
        
            for id in self.selected_clients:
               client = self.clients[id]
               data =self.testset[client.domain]
               loader = DataLoader(data, batch_size=512)
               client.train(round_idx, lr, model_dict)
               testacc=self.test_client(id,client.domain,loader)
               self.printer.info(f'the test acc of client {id} in {round_idx}/{self.global_rounds} is {testacc}')
               log_dict = {f'client_{id}_test_acc': testacc}
               self.logger.log(log_dict,step=round_idx)
               
            self.aggregate_models(round_idx)
            self.printer.info(f'the orginal aggregate of client acc is ')
            self.evaluated_new(round_idx)
            cuda.empty_cache() 
            
            #计算参数相似度
            cos_dict=self.calculate_similarty()
            self.printer.info(f'the parameters similarty of local and global  is {cos_dict}')
            # 修改 cos_dict，将键转换为字符串
            # cos_dict_str = {str(key): value for key, value in cos_dict.items()}
            # self.logger.log(cos_dict_str)
            
            
            for param in self.global_model.parameters():
                param.data.fill_(0)
            
            last_elements = [value[-1] for value in cos_dict.values()]
            self.aggregate_models_fc(round_idx,last_elements)
            
            if self.sBN:
                make_batchnorm_stats(self.batchnorm_dataset, self.global_model, self.device)
            
            final_acc=self.evaluate_fc(round_idx)
            log_dict = {f'final_test_acc': final_acc}
            self.logger.log(log_dict,step=round_idx)
            
            self.printer.info(f'{round_idx}/{self.global_rounds} cost: {(time.time() - st_time) / 60:.2f} min')
            self.printer.info('-' * 30)

    def calculate_similarty(self):
        cos_dict={}
        for id in self.selected_clients:
            cos_dict[id] = {}  # 为每个客户端创建一个空字典来存储相似度
            client = self.clients[id]
            client_model=client.model
            # 用一个列表存储每个客户端在每层的相似度
            similarity_list = []
            for i in range (6):
                ls1=client_model.get_layer_parameters(i+1)
                ls2=self.global_model.get_layer_parameters(i+1)
                result=calculate_average_cosine_similarity(ls1,ls2)
                similarity_list.append(result)
            cos_dict[id]=similarity_list
        return cos_dict


    def aggregate_models_fc(self, round_idx,ls):
        """
        get local models from selected clients and aggregate them
        """
        uploaded_models_avg = []
        
        weights_avg = []
        
        # 计算均值和方差
        mean_value = np.mean(ls)
        variance_value = np.var(ls)
        # 定义阈值
        threshold = mean_value +0.5*variance_value

        # 找出大于阈值的元素的索引
        indices = np.where(ls >= threshold)[0]
        
        
        for i in indices:
            id=self.selected_clients[i]
            client = self.clients[id]
            model = copy.deepcopy(client.model).to(self.device)
            uploaded_models_avg.append(model)
            if self.agg == 'uniform':
                weights_avg.append(1)
            elif self.agg == 'weighted':
                weights_avg.append(len(client.trainset))
            else:
                raise ValueError(f'invalid aggregation method: {self.agg}')
        wsum = sum(weights_avg)
        weights_avg = [i / wsum for i in weights_avg]
        self.aggregate(uploaded_models_avg, weights_avg)
        self.printer.info(f'the aggregate of high fc similarity client acc is ')
        self.evaluate_fc(round_idx)
        
    def evaluate_fc(self, round_idx):
        """
        Evaluate the accuracy of current model on each domain testset
        
        """
        st = time.time()
        acc_ls = []
        print_log = 'Evaluation: '
        for name in self.subset_list:
            data =self.testset[name]
            loader = DataLoader(data, batch_size=512)
            acc = self.test(name, loader)
            acc_ls.append(acc)
            if self.best_acc_fc.get(name):
                if acc > self.best_acc_fc[name]:
                    self.best_acc_fc[name] = acc
            else:
                self.best_acc_fc[name] = acc
            log_dict = {f'{name}_test_acc_of_better_fc': acc, f'{name}_best_acc_of_better_fc': self.best_acc_fc[name]}
            print_log += f'{name}: {acc:.2f}%; '
        
        avg_acc = sum(acc_ls) / len(acc_ls)
        best = False
        if self.best_acc_fc.get('global'):
            if avg_acc > self.best_acc_fc['global']:
                self.best_acc_fc['global'] = avg_acc
                best = True
        else:
            self.best_acc_fc['global'] = avg_acc
        
        log_dict = {f'global_test_acc_of_better_fc': avg_acc, f'global_best_acc_of_better_fc': self.best_acc_fc['global']}
        self.logger.log(log_dict, step=round_idx)

        print_log += f'avg_acc_of_better_fc: {avg_acc:.2f}%; best_acc_of_better_fc: {self.best_acc_fc["global"] :.2f}%'  
        self.printer.info(print_log)
        return avg_acc
        
        
class Client(ClientBase):
    def __init__(self, args, id, trainset) -> None:
        super(Client, self).__init__(args, id, trainset)


    
        
