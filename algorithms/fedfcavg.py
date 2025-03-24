import copy
import time
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm_
from algorithms.base import ClientBase, ServerBase
from utils import *



class FedFc(ServerBase):
    def __init__(self, args) -> None:
        super(FedFc, self).__init__(args, Client)
        self.clients_fc_weights = {}
        self.clients_fc_bias = {}
        self.aggregate_round=5
        
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
               client.train(round_idx,lr,self.clients_fc_weights,self.clients_fc_bias,model_dict,self.aggregate_round)
               data =self.testset[client.domain]
               loader = DataLoader(data, batch_size=512)
               testacc=self.test_client(id,client.domain,loader)
               self.printer.info(f'the test acc of client {id} in {round_idx}/{self.global_rounds} is {testacc}')
               log_dict = {f'client_{id}_test_acc': testacc}
               self.logger.log(log_dict,step=round_idx)
               
            self.aggregate_models(round_idx,self.aggregate_round)
            
            
            
    def aggregate_models(self, round_idx,aggregate_round):
        """
        聚合各客户端上传的fc层列表
        每个客户端持有一个fc层列表，需要对列表中相同位置的分类器进行聚合
        """
        # 更新各模型的参数分类器
        # 首先确定每个客户端的权重
        self.clients_fc_weights = {}  # 清空字典
        self.clients_fc_bias = {}
        for i, id in enumerate(self.selected_clients):
            client = self.clients[id]
            # 获取客户端模型
            client_model = client.model
                # 只收集fc层的参数
            self.clients_fc_weights[id] = client_model.fc.weight.data.clone()
            self.clients_fc_bias[id]=client_model.fc.bias.data.clone()   
        
        if((round_idx+1)%aggregate_round==0 and round_idx!=0):
            uploaded_models = []
            weights = []
            for i, id in enumerate(self.selected_clients):
                client = self.clients[id]
            
                model = copy.deepcopy(client.model).to(self.device)
                uploaded_models.append(model)
                if self.agg == 'uniform':
                    weights.append(1)
                elif self.agg == 'weighted':
                    weights.append(len(client.trainset))
                else:
                    raise ValueError(f'invalid aggregation method: {self.agg}')
            wsum = sum(weights)
            weights = [i / wsum for i in weights]
            self.aggregate(uploaded_models, weights)
            final_acc=self.evaluated_new(round_idx)
            log_dict = {f'final_test_acc': final_acc}
            self.logger.log(log_dict,step=round_idx)
        
        
    
       
            
    
    

class Client(ClientBase):
    def __init__(self, args, id, trainset) -> None:
        super(Client, self).__init__(args, id, trainset)
        self.fc_weights=self.model.fc.weight.data.clone()
        self.fc_bias  = self.model.fc.bias.data.clone()
        
    def set_parameters(self, state_dict):
        self.printer.debug(f'set parameters')
        # 保存当前的fc层参数
        current_fc_weight = self.model.fc.weight.data.clone()
        current_fc_bias = self.model.fc.bias.data.clone()

        # 加载全局模型参数
        self.model.load_state_dict(state_dict)

        # 恢复本地fc层参数
        self.model.fc.weight.data = current_fc_weight
        self.model.fc.bias.data = current_fc_bias
    
    def prepare(self, lr, state_dict):
        self.printer.debug(f'client preparation')
        self.model.to(self.device)
        self.set_parameters(state_dict)
        for group in self.optimizer_dict['param_groups']:
            group['lr'] = lr
        self.optimizer.load_state_dict(self.optimizer_dict)


    def train(self, round_idx, lr,clients_fc_weights,clients_fc_bias,model_dict,aggregate_round):
        for group in self.optimizer_dict['param_groups']:
            group['lr'] = lr
        if(round_idx%aggregate_round==0 and round_idx!=0):
            self.prepare(lr, model_dict)
        self.optimizer.load_state_dict(self.optimizer_dict)
        self.model.train(True)
        loss_meter = AverageMeter()
        local_steps=self.local_steps+50
        if(round_idx>0):
            local_steps=self.local_steps
            copied_dict_weights = copy.deepcopy(clients_fc_weights)
            copied_dict_bias = copy.deepcopy(clients_fc_bias)
            del copied_dict_weights[self.id]
            del copied_dict_bias[self.id]
        
        for step in range(local_steps):
            loss_meter.reset()
            if(round_idx>0 and self.local_steps-step>=2):
                if copied_dict_weights:
                    random_key = random.choice(list(copied_dict_weights.keys()))
                    # 获取对应的元素（可选）
                    selected_weight = clients_fc_weights[random_key]
                    selected_bias = clients_fc_bias[random_key]
                
                    # 将选择的FC层参数加载到固定的FC层
                    fixed_fc_weight = selected_weight.clone().detach()  # 分离计算图，防止梯度回传
                    fixed_fc_bias = selected_bias.clone().detach()
                for param in self.model.parameters():
                    param.requires_grad = True
                #双头分类器
                for data in self.loader:
                    x, y = data['x'].to(self.device), data['y'].to(self.device)
                    if len(y) < 2:
                        continue
                    logits = self.model(x)
                    client_loss = self.ls_func(logits, y)
                    features=self.model.get_feature(x)
                    
                    fixed_logits = F.linear(features, fixed_fc_weight, fixed_fc_bias)
                    fixed_loss = self.ls_func(fixed_logits, y)
                    loss=0.5*client_loss+0.5*fixed_loss
                    self.optimizer.zero_grad()
                    loss.backward()
                    clip_grad_norm_(self.model.parameters(), self.clip_grad)
                    self.optimizer.step()
                    loss_meter.update(loss.item(), y.size(0))
                self.printer.info(f'C{self.id:<2d}:{step}/{self.local_steps} >> avg loss {loss_meter.avg:.2f}') 
                
                
            elif(round_idx>0):
                # 在模型初始化后冻结参数
                for param in self.model.parameters():
                    param.requires_grad = False
                # 单独解冻全连接层
                for param in self.model.fc.parameters():
                    param.requires_grad = True   
                         
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
                
              
            else:
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
