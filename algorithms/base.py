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
from utils import get_optimizer, get_net_builder
from datasets import fetch_dataset, split_dataset, SubDataset
from utils import AverageMeter, make_batchnorm_stats

class ServerBase(object):
    def __init__(self, args, Client) -> None:
        self.args = args
        self.net = args.net
        self.algorithm = args.algorithm
        self.device = torch.device('cuda:0')
        self.global_rounds = args.global_rounds
        self.current_round = 0
        self.num_clients = args.num_clients
        self.num_join_clients = int(self.num_clients * args.join_ratio)
        self.clients = []
        self.logger = args.logger    #wandb跟踪器
        self.printer = args.printer    #日志打印器
        self.save_dir = args.save_dir
        self.exp_tag = args.exp_tag     #实验运行环境记录tag
        self.load_path = args.load_path   
        self.data_shape = args.data_shape
        self.num_classes = args.num_classes
        self.subset_list = args.subset_list   #数据子集列表
        self.clip_grad = args.clip_grad
        self.selection = None      
        self.agg = args.agg             #client加权方式
        self.best_acc = {}
        self.selected_clients = []
        self.global_model = self.make_model()   
        self.global_optimizer = get_optimizer(self.global_model, optim_name='SGD', lr=0.05, momentum=args.global_momentum, weight_decay=0, nesterov=False)
        self.optimizer = get_optimizer(self.global_model, optim_name=args.optim, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.global_rounds, eta_min=0)
        self.clt_set, self.testset = self.make_dataset(args)
        self.make_client(args, Client)
        self.sBN = args.sBN
        if self.sBN:
            self.batchnorm_dataset = self.make_norm_dataset()

    def make_model(self):
        self.printer.debug(f'make model: {self.net}')
        net_builder = get_net_builder(self.net)   #获得网络构造器
        model = net_builder(self.data_shape, self.num_classes).to(self.device)
        return model    
    
    def make_dataset(self, args):
        self.printer.debug('make dataset')
        datasets = fetch_dataset(args.data_dir, args.dataset, train=True)
        testset = fetch_dataset(args.data_dir, args.dataset, train=False)
        client_per_domain = self.num_clients // len(self.subset_list)
        clt_set = {}
        for dm in self.subset_list:
            dm_set = datasets[dm]
            data_idx = split_dataset(dm_set, args, client_per_domain)
            clt_set[dm] = (dm_set, data_idx)
        return clt_set, testset
    
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
    
    def make_norm_dataset(self):
        self.printer.debug(f'make norm dataset')
        dset = {}
        for name in self.subset_list:
            dset[name] = self.clt_set[name][0]
        return dset

    def make_client(self, args, clientObj):
        self.printer.debug('make clients')
        num_domain = len(self.subset_list)
        client_per_domain = self.num_clients // num_domain
        matrix = np.zeros((self.num_clients, self.num_classes), dtype=np.int32)
        for i in range(self.num_clients):
            domain = self.subset_list[i // client_per_domain]
            dataset, data_idx = self.clt_set[domain]
            idx_i = data_idx[i % client_per_domain]
            targets = dataset.targets
            idx_i = np.array(idx_i, dtype=np.int32)
            for j in range(self.num_classes):
                matrix[i, j] = np.sum(targets[idx_i] == j)
            client_set = SubDataset(dataset, idx_i)
            client = clientObj(args, i, client_set)
            self.clients.append(client)

        self.statistics = matrix
        self.printer.info(f'----------------label distribution of clients-----------------\n{matrix}')


    def select_clients(self, round_idx):
        return list(np.random.choice(self.num_clients, self.num_join_clients, replace=False))
        # if self.selection is None:
        #     self.selection = [list(np.random.choice(self.num_clients, self.num_join_clients, replace=False)) for i in range(self.global_rounds)]
        # return self.selection[round_idx]
    
    
    def aggregate(self, uploaded_models, weights): 
        """
        aggregate model parameters based on the uploaded weights;
        """   
        self.printer.debug("------Base aggregate_models------")
        if len(uploaded_models) > 0:
            self.printer.info(f'aggregation weights: {weights}')
            with torch.no_grad():
                shadow_model = copy.deepcopy(self.global_model)
                for param in shadow_model.parameters():
                    param.data.zero_()
                
                for w, client_model in zip(weights, uploaded_models):
                    for new_param, param in zip(client_model.parameters(), shadow_model.parameters()):
                        param.data += w * new_param.data.clone()
                
                 # Directly replace global model parameters with the aggregated ones
                with torch.no_grad():
                    for new_param, param in zip(shadow_model.parameters(), self.global_model.parameters()):
                        param.data.copy_(new_param.data)  # Directly replace parameter data

                # update batchnorm statistics
                for i in range(len(uploaded_models)):
                    w, client_model = weights[i], uploaded_models[i]
                    for gmodule, lmodule in zip(self.global_model.modules(), client_model.modules()):
                        if isinstance(gmodule, nn.BatchNorm2d):
                            if i == 0:
                                gmodule.running_mean = lmodule.running_mean.clone() * w
                                gmodule.running_var = lmodule.running_var.clone() * w
                            else:
                                gmodule.running_mean += lmodule.running_mean.clone() * w
                                gmodule.running_var += lmodule.running_var.clone() * w
        else:
            self.printer.info('no uploaded models, skip aggregation')
    
    def aggregate_models(self, round_idx):
        """
        get local models from selected clients and aggregate them
        """
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
            
    
    @torch.no_grad()
    def test(self, data_name, loader):
        self.printer.debug(f'-----------------testing on {data_name}-----------------')
        all_y, all_logits = [], []
        self.global_model.train(False)
        self.global_model.to(self.device)
        for data in loader:   
            x, y = data['x'].to(self.device), data['y'].to(self.device)
            logits = self.global_model(x)
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
               client.train(round_idx, lr, model_dict)
               data =self.testset[client.domain]
               loader = DataLoader(data, batch_size=512)
               testacc=self.test_client(id,client.domain,loader)
               self.printer.info(f'the test acc of client {id} in {round_idx}/{self.global_rounds} is {testacc}')
               log_dict = {f'client_{id}_test_acc': testacc}
               self.logger.log(log_dict,step=round_idx)
               
            self.aggregate_models(round_idx)
            cuda.empty_cache() 
            if self.sBN:
                make_batchnorm_stats(self.batchnorm_dataset, self.global_model, self.device)
            final_acc=self.evaluated_new(round_idx)
            log_dict = {f'final_test_acc': final_acc}
            self.logger.log(log_dict,step=round_idx)
            
            self.printer.info(f'{round_idx}/{self.global_rounds} cost: {(time.time() - st_time) / 60:.2f} min')
            self.printer.info('-' * 30)
    
    def evaluate(self, round_idx):
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
            if self.best_acc.get(name):
                if acc > self.best_acc[name]:
                    self.best_acc[name] = acc
            else:
                self.best_acc[name] = acc
            log_dict = {f'{name}_test_acc': acc, f'{name}_best_acc': self.best_acc[name]}
            self.logger.log(log_dict, step=round_idx)
            print_log += f'{name}: {acc:.2f}%; '
        
        avg_acc = sum(acc_ls) / len(acc_ls)
        best = False
        if self.best_acc.get('global'):
            if avg_acc > self.best_acc['global']:
                self.best_acc['global'] = avg_acc
                best = True
        else:
            self.best_acc['global'] = avg_acc
        
        log_dict = {f'global_test_acc': avg_acc, f'global_best_acc': self.best_acc['global']}
        self.logger.log(log_dict, step=round_idx)

        print_log += f'avg_acc: {avg_acc:.2f}%; best_acc: {self.best_acc["global"] :.2f}%'
        self.save_model(round_idx, best=best)
        self.printer.info(print_log)
        self.printer.info(f'evaluation cost: {(time.time() - st)/60:.2f} min')

    def evaluated_new(self, round_idx):
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
            if self.best_acc.get(name):
                if acc > self.best_acc[name]:
                    self.best_acc[name] = acc
            else:
                self.best_acc[name] = acc
            log_dict = {f'{name}_test_acc': acc, f'{name}_best_acc': self.best_acc[name]}
            self.logger.log(log_dict, step=round_idx)
            print_log += f'{name}: {acc:.2f}%; '
        
        avg_acc = sum(acc_ls) / len(acc_ls)
        best = False
        if self.best_acc.get('global'):
            if avg_acc > self.best_acc['global']:
                self.best_acc['global'] = avg_acc
                best = True
        else:
            self.best_acc['global'] = avg_acc
        
        log_dict = {f'global_test_acc': avg_acc, f'global_best_acc': self.best_acc['global']}
        self.logger.log(log_dict, step=round_idx)

        print_log += f'avg_acc: {avg_acc:.2f}%; best_acc: {self.best_acc["global"] :.2f}%'
        self.save_model(round_idx, best=best)
        self.printer.info(print_log)
        self.printer.info(f'evaluation cost: {(time.time() - st)/60:.2f} min')
        return avg_acc
        
    def save_model(self, round, best=False):
        save_path = os.path.join(self.save_dir, self.exp_tag)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if best:
            path = os.path.join(save_path, f'{self.net}_models_best.pth')
            ckpt = {
                'ckpt_model': self.global_model.state_dict(),
            }
            torch.save(ckpt, path)


class ClientBase(object):
    def __init__(self, args, id, trainset) -> None:
        self.id = id
        self.args = args
        self.net = args.net
        self.exp_tag = args.exp_tag
        self.load_path = args.load_path
        self.global_rounds = args.global_rounds
        self.local_steps = args.c_steps
        self.save_dir = args.save_dir
        self.trainset = trainset
        self.data_shape = args.data_shape
        self.num_classes = args.num_classes
        self.logger = args.logger
        self.printer = args.printer
        self.clip_grad = args.clip_grad
        self.num_samples = len(trainset)
        self.device = torch.device('cuda:0')
        self.model = self.make_model()
        self.optimizer = get_optimizer(self.model, optim_name=args.optim, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        self.optimizer_dict = self.optimizer.state_dict()
        self.loader = DataLoader(trainset, batch_size=args.c_bs, shuffle=True)
        self.ls_func = nn.CrossEntropyLoss()
        self.local_round=0
        client_per_domain=self.num_classes//len(args.subset_list)
        self.domain=args.subset_list[id//client_per_domain]
        

    def make_model(self):
        self.printer.debug(f'make model: {self.net}')
        net_builder = get_net_builder(self.net)
        model = net_builder(self.data_shape, self.num_classes).to(self.device)
        return model   

    def set_parameters(self, state_dict):
        self.printer.debug(f'set parameters')
        self.model.load_state_dict(state_dict)
    
    def prepare(self, lr, state_dict):
        self.printer.debug(f'client preparation')
        self.model.to(self.device)
        self.set_parameters(state_dict)
        for group in self.optimizer_dict['param_groups']:
            group['lr'] = lr
        self.optimizer.load_state_dict(self.optimizer_dict)

    def train(self, round_idx, lr, state_dict):
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
        self.model.to('cpu')

