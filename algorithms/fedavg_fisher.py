# This file contains the implementation of FedAvg algorithm
import copy
import time
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm_
from algorithms.baseplus import ClientBase, ServerBase
from utils import *
from nngeometry.metrics import FIM
from nngeometry.object import PMatDiag

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

''' Fisher版本'''
class FedAvgFisher(ServerBase):
    def __init__(self, args) -> None:
        super(FedAvgFisher, self).__init__(args, Client)
        self.best_acc_fc={}
        self.best_acc_fisher={}
        
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
            # # 修改 cos_dict，将键转换为字符串
            # cos_dict_str = {str(key): value for key, value in cos_dict.items()}
            # self.logger.log(cos_dict_str)
            
            
            for param in self.global_model.parameters():
                param.data.fill_(0)
            
            last_elements = [value[-1] for value in cos_dict.values()]
            self.aggregate_models_fc(round_idx,last_elements)
            
            self.aggregate_models_fisher(round_idx,last_elements)
            
            if self.sBN:
                make_batchnorm_stats(self.batchnorm_dataset, self.global_model, self.device)
            
            final_acc=self.evaluate_fisher(round_idx)
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

    def aggregate_models_fisher(self, round_idx,ls):
        """
        get local models from selected clients and aggregate them
        """
        uploaded_models_fisher=[]
        weights_fisher=[]
        fisher=[]
        # 计算均值和方差
        mean_value = np.mean(ls)
        variance_value = np.var(ls)
        # 定义阈值
        threshold = mean_value + variance_value

        
        # 找出阈值的元素的索引
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
        self.fisher_gradient_descent(uploaded_models_fisher, weights_fisher,fisher,round_idx)
        
    def fisher_gradient_descent(self, uploaded_models_fisher, weights, fisher_matrices, round_idx,learning_rate=0.05):
        """
        使用客户端的Fisher矩阵对全局模型进行梯度下降
        
        参数:
            uploaded_models_fisher: 上传的客户端模型列表
            weights: 客户端权重列表
            fisher_matrices: 客户端的Fisher矩阵列表
            learning_rate: 学习率
        """
        learning_rate = max(0.001, learning_rate * (0.5 ** round_idx))
        with torch.no_grad():
            # 获取全局模型的状态字典
            global_state_dict = self.global_model.state_dict()
            
            # 获取所有参数的名称列表
            param_names = [name for name, _ in self.global_model.named_parameters()]
            
            # 对每个参数进行更新
            for key in global_state_dict.keys():
                if 'num_batches_tracked' in key or key not in param_names:
                    # 跳过batch normalization统计数据和非参数张量
                    continue
                    
                # 初始化梯度累加器
                gradient_acc = torch.zeros_like(global_state_dict[key])
                fisher_acc = torch.zeros_like(global_state_dict[key])
                
                # 获取当前参数在参数列表中的索引
                param_idx = param_names.index(key)
                
                # 对每个客户端模型计算梯度
                for client_idx, client_model in enumerate(uploaded_models_fisher):
                    client_params = client_model.state_dict()[key]
                    
                    # 获取对应参数的Fisher信息
                    client_fisher = fisher_matrices[client_idx][param_idx]
                    
                    # 将Fisher值转换为与参数相同形状的张量
                    fisher_tensor = torch.full_like(client_params, client_fisher)
                    
                    # 计算参数差异（梯度）
                    param_diff = client_params - global_state_dict[key]
                    
                    # 使用Fisher矩阵和权重加权梯度
                    weighted_grad = param_diff * fisher_tensor * weights[client_idx]
                    gradient_acc.add_(weighted_grad)
                    fisher_acc.add_(fisher_tensor * weights[client_idx])
                
                # 归一化梯度并应用学习率
                normalized_gradient = gradient_acc / (fisher_acc + 1e-10)
                global_state_dict[key].add_(normalized_gradient * learning_rate)
                
            # 更新全局模型参数
            self.global_model.load_state_dict(global_state_dict)
      
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
        return avg_acc
          
class Client(ClientBase):
    def __init__(self, args, id, trainset) -> None:
        super(Client, self).__init__(args, id, trainset)
        self.global_model=self.make_model()
        
    def train(self, round_idx, lr, state_dict):
        self.prepare(lr, state_dict)
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
                loss = self.ls_func(logits, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_meter.update(loss.item(), y.size(0))
            self.printer.info(f'C{self.id:<2d}:{step}/{self.local_steps} >> avg loss {loss_meter.avg:.2f}')
            
        
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
      
