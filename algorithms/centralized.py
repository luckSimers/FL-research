import os
import time
import torch
import logging
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from torch.nn.utils.clip_grad import clip_grad_norm_
from utils import get_optimizer, get_net_builder
from datasets import fetch_dataset, split_dataset, SubDataset
from utils import AverageMeter, mixup_data, make_batchnorm_stats

class Centralized(object):

    def __init__(self, args) -> None:
        self.algorithm = args.algorithm
        self.best_acc = {}
        self.current_round = 0
        self.dataset = args.dataset
        self.device = torch.device('cuda:0')
        self.exp_tag = args.exp_tag
        self.global_rounds = args.global_rounds
        self.logger = args.logger
        self.printer = args.printer
        self.load_path = args.load_path
        self.data_shape = args.data_shape
        self.net = args.net
        self.save_dir = args.save_dir
        self.subset_list = args.subset_list
        self.clip_grad = args.clip_grad
        self.num_classes = args.num_classes
        self.train_set, self.testset = self.make_dataset(args)
        self.model = self.make_model()
        self.optimizer = get_optimizer(self.model, optim_name=args.optim, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.global_rounds, eta_min=0)        
        self.loss = nn.CrossEntropyLoss()
        self.train_loader = DataLoader(self.train_set, batch_size=128, shuffle=True, drop_last=True)


    def train(self, round_idx):
        st = time.time()
        self.model.train(True)
        for data in self.train_loader:
            x, y = data['x'].to(self.device), data['y'].to(self.device)
            logits = self.model(x)
            loss = self.loss(logits, y)
            self.optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(self.model.parameters(), self.clip_grad)
            self.optimizer.step()
        self.printer.info(f'>> training cost {(time.time() - st)/60:.2f} min')

    def make_model(self):
        self.printer.debug(f'make model: {self.net}')
        net_builder = get_net_builder(self.net)
        model = net_builder(self.data_shape, self.num_classes).to(self.device)
        return model    
        
    def make_dataset(self, args):
        self.printer.debug('make dataset')
        datasets = fetch_dataset(args.data_dir, args.dataset, train=True)
        testset = fetch_dataset(args.data_dir, args.dataset, train=False)
        return datasets[self.dataset], testset
        
    def run(self):
        """
        in this function, server only:
            1.send model to clients;
            2.select active clients for local training;
            3.receive models from clients;
            4.aggregate models;
        rewrite this function if you wanna use a more complex algorithm
        """
        for round_idx in range(self.global_rounds):
            self.printer.info(f'Round {round_idx}/{self.global_rounds}...')
            st_time = time.time()
            self.train(round_idx)
            self.scheduler.step()
            self.evaluate(round_idx)
            self.printer.info(f'time cost {(time.time() - st_time)/60:.2f} min')
    

    @torch.no_grad()
    def test(self, data_name, loader):
        self.printer.debug(f'-----------------testing on {data_name}-----------------')
        all_y, all_logits = [], []
        self.model.train(False)
        self.model.to(self.device)
        for data in loader:   
            x, y = data['x'].to(self.device), data['y'].to(self.device)
            logits = self.model(x)
            all_y.append(y)
            all_logits.append(logits)
        y = torch.cat(all_y, dim=0)
        logits = torch.cat(all_logits, dim=0)
        test_acc = (y == torch.argmax(logits, dim=1)).float().mean().item() * 100
        return test_acc

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


    def save_model(self, round, best=False):
        save_path = os.path.join(self.save_dir, self.exp_tag)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if best:
            path = os.path.join(save_path, f'{self.net}_models_best.pth')
            ckpt = {
                'ckpt_model': self.model.state_dict(),
            }
            torch.save(ckpt, path)
        else:
            path = os.path.join(save_path, f'{self.net}_{round}.pth')
            ckpt = {
                'ckpt_model': self.model.state_dict(),
            }
            torch.save(ckpt, path) 


    @staticmethod
    def get_argument():
        return [
        ]