# This file contains the implementation of FedAvg algorithm
import copy
import time
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm_
from algorithms.baseplus import ClientBasePlus, ServerBasePlus
from utils import *
from nngeometry.metrics import FIM
from nngeometry.object import PMatDiag

'''KD Fisher版本'''
class FedAvgLayer(ServerBasePlus):
    def __init__(self, args) -> None:
        super(FedAvgLayer, self).__init__(args, Client)
    

class Client(ClientBasePlus):
    def __init__(self, args, id, trainset) -> None:
        super(Client, self).__init__(args, id, trainset)
        self.global_model=self.make_model()
        
    def train_withfc_fisher(self, round_idx, lr, state_dict):
        self.printer.debug(f'set parameters')
        self.global_model.load_state_dict(state_dict)
        self.model.train(True)
        loss_meter = AverageMeter()
        loss_meter_classfication = AverageMeter()
        loss_meter_KD = AverageMeter()
        if(round_idx%3==0):
            self.local_steps-=10
        for step in range(self.local_steps):
            loss_meter.reset()
            for data in self.loader:
                x, y = data['x'].to(self.device), data['y'].to(self.device)
                if len(y) < 2:
                    continue
                logits = self.model(x)
                loss = self.ls_func(logits, y)
                loss_meter_classfication.update(loss.item(), y.size(0))
                if(round_idx>0):
                    feature_global=self.global_model.get_feature_layer(x,round_idx)
                    feature_local=self.model.get_feature_layer(x,round_idx)
                    loss2=nn.MSELoss()(feature_local, feature_global)
                    loss_meter_KD.update(loss2.item(), y.size(0))
                    loss=loss+loss2
                self.optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(self.model.parameters(), self.clip_grad)
                self.optimizer.step()
                loss_meter.update(loss.item(), y.size(0))
            self.printer.info(f'C{self.id:<2d}:{step}/{self.local_steps} >> avg loss {loss_meter.avg:.2f}>>classfication loss {loss_meter_classfication.avg:.2f}>>KD loss{loss_meter_KD.avg:.2f}')
            
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
