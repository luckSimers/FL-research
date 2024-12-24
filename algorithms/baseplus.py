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


class ServerBasePlus(ServerBase):
    def __init__(self, args, Client) -> None:
       super(ServerBasePlus, self).__init__(args, Client)
       self.best_acc_fc={}
       self.best_acc_fisher={}
       
    @torch.no_grad()
    def test_client(self,id,data_name, loader):
        self.printer.debug(f'-----------------testing on {data_name}-----------------')
        all_y, all_logits = [], []
        model=self.clients[id].model
        model.train(False)
        model.to(self.device)
        for data in loader:   
            x, y = data['x'].to(self.device), data['y'].to(self.device)
            logits = model(x)
            all_y.append(y)
            all_logits.append(logits)
        y = torch.cat(all_y, dim=0)
        logits = torch.cat(all_logits, dim=0)
        test_acc = (y == torch.argmax(logits, dim=1)).float().mean().item() * 100
        return test_acc
        
    def run(self):
        if self.sBN:
            make_batchnorm_stats(self.batchnorm_dataset, self.global_model, self.device)
        for round_idx in range(self.global_rounds):
            st_time = time.time()
            self.selected_clients = self.select_clients(round_idx)
            self.selected_clients.sort()
            lr = self.scheduler.get_last_lr()[0]
            model_dict = self.global_model.state_dict()

            for id in self.selected_clients:
               client = self.clients[id]
               client.train_withfc_fisher(round_idx, lr, model_dict)
               
               data =self.testset[client.domain]
               loader = DataLoader(data, batch_size=512)
               testacc=self.test_client(id,client.domain,loader)
               self.printer.info(f'the test acc of client {id} in {round_idx}/{self.global_rounds} is {testacc}')
               log_dict = {f'client_{id}_test_acc': testacc}
               self.logger.log(log_dict)
               
            self.aggregate_models(round_idx)
            self.printer.info(f'the orginal aggregate of client acc is ')
            self.evaluate(round_idx)
            cuda.empty_cache() 
            cos_dict=self.calculate_similarty()
            self.printer.info(f'the parameters of local and global  is {cos_dict}')
            for param in self.global_model.parameters():
                param.data.fill_(0)
            last_elements = [value[-1] for value in cos_dict.values()]
            self.aggregate_models_fisher(round_idx,last_elements)
            if self.sBN:
                make_batchnorm_stats(self.batchnorm_dataset, self.global_model, self.device)
            self.evaluate_fisher(round_idx)
            self.printer.info(f'{round_idx}/{self.global_rounds} cost: {(time.time() - st_time) / 60:.2f} min')
            self.printer.info('-' * 30)
    
    def aggregate_fisher(self, uploaded_models_fisher, weights,fisher):
        """
        Aggregate model parameters based on Fisher information weighting.
        
        """
        self.printer.debug("------Base aggregate_models------")
        
        if not uploaded_models_fisher:
            self.printer.info('no uploaded models, skip aggregation')
            return
        # 清空全局优化器的梯度
        self.global_optimizer.zero_grad()
        self.printer.info(f'Aggregation weights (Fisher-based): {weights}')
        # 初始化Fisher加权梯度字典
        # 假设全局参数展平为 w
        global_params = list(self.global_model.parameters())
        w = torch.cat([p.data.view(-1) for p in global_params])
         # Initialize the gradient to zero
        grad = torch.zeros_like(w)
        for weight, (model, fisher_information) in zip(weights, zip(uploaded_models_fisher, fisher)):
            model_params = list(model.parameters())
            w_local = torch.cat([p.data.view(-1) for p in model_params])
            grad+=weight*fisher_information*(w-w_local)
          # Apply the aggregated gradient to the global model
        for p, g in zip(self.global_model.parameters(), grad.split([p.numel() for p in self.global_model.parameters()])):
            p.grad = g.view_as(p)
        
        # Optionally, perform a step of gradient descent if needed
        self.global_optimizer.step()
        
        print("Global model updated using Fisher-based gradient step.")

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
            self.logger.log(log_dict, step=round_idx)
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
              
    def evaluate_fisher(self, round_idx):
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
            if self.best_acc_fisher.get(name):
                if acc > self.best_acc_fisher[name]:
                    self.best_acc_fisher[name] = acc
            else:
                self.best_acc_fisher[name] = acc
            log_dict = {f'{name}_test_acc_of_fisher': acc, f'{name}_best_acc_of_fisher': self.best_acc_fisher[name]}
            self.logger.log(log_dict, step=round_idx)
            print_log += f'{name}: {acc:.2f}%; '
        
        avg_acc = sum(acc_ls) / len(acc_ls)
        best = False
        if self.best_acc_fisher.get('global'):
            if avg_acc > self.best_acc_fisher['global']:
                self.best_acc_fisher['global'] = avg_acc
                best = True
        else:
            self.best_acc_fisher['global'] = avg_acc
        
        log_dict = {f'global_test_acc_of_fisher': avg_acc, f'global_best_acc_of_fisher': self.best_acc_fisher['global']}
        self.logger.log(log_dict, step=round_idx)

        print_log += f'avg_acc_of_fisher: {avg_acc:.2f}%; best_acc_of_fisher: {self.best_acc_fisher["global"] :.2f}%'    
        self.printer.info(print_log)
    def aggregate_models_fisher(self, round_idx,ls):
        """
        get local models from selected clients and aggregate them
        """
        uploaded_models_avg = []
        uploaded_models_fisher=[]
        weights_avg = []
        weights_fisher=[]
        fisher=[]
        # 计算均值和方差
        mean_value = np.mean(ls)
        variance_value = np.var(ls)
        # 定义阈值
        threshold = mean_value + variance_value

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
        # 找出大于阈值的元素的索引
        indices = np.where(ls <= threshold)[0]
        
        for i in indices:
            id=self.selected_clients[i]
            client = self.clients[id]
            model = copy.deepcopy(client.model).to(self.device)
            fisher.append(client.fisher)
            uploaded_models_fisher.append(model)
            if self.agg == 'uniform':
                weights_fisher.append(1)
            elif self.agg == 'weighted':
                weights_fisher.append(len(client.trainset))
            else:
                raise ValueError(f'invalid aggregation method: {self.agg}')
        wsum = sum(weights_fisher)
        weights_fisher = [i / wsum for i in weights_fisher]
        self.aggregate_fisher(uploaded_models_fisher, weights_fisher,fisher)
    
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
      


class ClientBasePlus(ClientBase):
    def __init__(self, args, id, trainset) -> None:
      super(ClientBasePlus, self).__init__(args, id, trainset)
      client_per_domain=self.num_classes//len(args.subset_list)
      self.domain=args.subset_list[id//client_per_domain]
      self.fisher=None
      
    def train_withfc_fisher(self, round_idx, lr, state_dict):
        self.prepare(lr, state_dict)
        self.model.train(True)
        loss_meter = AverageMeter()
        for step in range(self.local_steps):
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
            
        torch.cuda.empty_cache()  # 清理缓存，释放GPU内存
        self.model.eval()
        for batch in self.loader:
            # 将数据和标签移到GPU
            x_batch = batch['x'].to('cuda', non_blocking=True)  # 数据移到GPU
            y_batch = batch['y'].to('cuda', non_blocking=True)  # 标签移到GPU

             # 这里可以进行其他的处理，例如Fisher信息矩阵计算
        fisher_batch = FIM(model=self.model,
                                loader=(x_batch, y_batch),  # 注意是元组形式
                                representation=PMatDiag,
                                device='cuda',
                                n_output=self.num_classes)
        self.fisher=fisher_batch.get_diag()

        self.model.to('cpu')  # 计算完后将模型移回CPU
        torch.cuda.empty_cache()  # 清理缓存，释放GPU内存
        

   

