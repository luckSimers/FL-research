# This file contains the implementation of FedAvg algorithm
import copy
import time
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm_
from algorithms import ClientBase, ServerBase
from utils import *



class FedBN(ServerBase):
    def __init__(self, args) -> None:
        super(FedBN, self).__init__(args, Client)
    

class Client(ClientBase):
    def __init__(self, args, id, trainset) -> None:
        super(Client, self).__init__(args, id, trainset)
        
    def set_parameters(self, state_dict):
        self.printer.debug(f'set parameters')
        
        # 过滤出非BatchNorm层的参数
        filtered_state_dict = {
            k: v for k, v in state_dict.items() if 'bn' not in k.lower()
        }
        self.model.load_state_dict(filtered_state_dict, strict=False)

        if self.local_round != 0:
            save_path = os.path.join(self.save_dir, "client" + str(self.id))
            file_path = os.path.join(save_path, f'model_round_{(self.local_round-1)}.pt')

            # 加载BN层参数
            if os.path.exists(file_path):
                self.printer.debug(f'Loading BN parameters from {file_path}')
                bn_state_dict = torch.load(file_path)['ckpt_model']

                # 过滤出BN层参数
                filtered_bn_state_dict = {
                    k: v for k, v in bn_state_dict.items() if 'bn' in k.lower()
                }

                # 更新模型的BN层参数
                self.model.load_state_dict(filtered_bn_state_dict, strict=False)
                self.printer.info(f'BN parameters loaded for round {self.local_round}')
            else:
                self.printer.warning(f'File not found: {file_path}')
                
    def save_model(self, round):
        save_path = os.path.join(self.save_dir, "client" + str(self.id))
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # 仅保存BatchNorm层的参数
        bn_state_dict = {
            k: v for k, v in self.model.state_dict().items() if isinstance(self.model._modules[k.split('.')[0]], nn.BatchNorm2d)
        }
        path = os.path.join(save_path, f'{self.net}_models_best.pth')

        # 保存BN层的状态字典
        file_path = os.path.join(save_path, f'model_round_{self.local_round}.pt')
        torch.save({'ckpt_model': bn_state_dict}, file_path)

        self.printer.info(f'BN parameters saved to {file_path}')

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
            self.save_model(round_idx)
            self.local_round+=1
        
