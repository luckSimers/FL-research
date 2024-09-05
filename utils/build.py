# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import math
import logging
import random
import torch.nn as nn
import torch.distributed as dist
from torch.optim import SGD, AdamW
from torch.utils.data import DataLoader
from datasets import FetchData, Subset4FL
import nets as models
from nets.utils import param_groups_layer_decay, param_groups_weight_decay

def get_net_builder(name):
    '''
    return a function that builds a model
    Args:
        name: model name
    '''
    model_dict = sorted(name for name in models.__dict__ 
                        if callable(models.__dict__[name]))
    if name in model_dict:
        return models.__dict__[name]
    else:
        raise Exception(f'No model named {name},\nexpected: {model_dict},\nreceived: {name}')
    

class BaseHeadSplit(nn.Module):
    def __init__(self, base, head, **kwargs):
        super(BaseHeadSplit, self).__init__()

        self.base = base
        self.head = head

        for k, v in kwargs.items():
            setattr(self, k, v)
        
    def forward(self, x):
        out = self.base(x)
        out = self.head(out)

        return out
    
class TwoHead(nn.Module):
    def __init__(self, base, **kwargs):
        super(TwoHead, self).__init__()

        self.base = base
        fet_dim = base.fc.in_features
        num_class = base.fc.out_features
        self.base.fc = nn.Identity()
        self.fc = nn.Linear(fet_dim, num_class)

        self.fc_aux = nn.Sequential(
            nn.Linear(fet_dim, fet_dim // 4),
            nn.BatchNorm1d(fet_dim // 4, track_running_stats=False),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            
            nn.Linear(fet_dim // 4, fet_dim // 4),
            nn.BatchNorm1d(fet_dim // 4, track_running_stats=False),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            
            nn.Linear(fet_dim // 4, num_class)
        )

    def get_fet(self, x):
        return self.base(x)
    
    def forward(self, x):
        fet = self.base(x)
        logits = self.fc(fet)
        if self.training:
            logits_aux = self.fc_aux(fet)
            return logits, logits_aux

        return logits


def fetch_fl_dataset(dataset_name, data_root, partition_type='iid', client_num=1, ft_data=0):
    """
    fetch dataset
    return:
        {
        'train_ds',
        'test_ds',
        'local_data_num',
        'class_num',
        'local_distribution',
        'client_idx',
    }
    """
    dataset = FetchData(dataset_name, data_root, partition_type, client_num, ft_data=ft_data)
    train_ds, test_ds, local_data_num, class_num, other_params = dataset.load_data()
    data_info = {
        'train_ds': train_ds,
        'test_ds': test_ds,
        'local_data_num': local_data_num,
        'class_num': class_num,
        'local_distribution': other_params['local_counts'],
        'client_idx': other_params['client_idx'],
        'ft_idx': other_params['ft_idx'],
    }
    return data_info


def get_fl_loader(dataset, data_idx, batch_size, shuffle=False, transforms=None, drop_last=False):
    """
    get federated learning data loader
    """
    if transforms is None:
        transforms = []
    subset = Subset4FL(dataset, data_idx, transform=transforms)
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)


def get_optimizer(net, optim_name='SGD', lr=0.1, momentum=0.9, weight_decay=0, layer_decay=1.0, nesterov=True, bn_wd_skip=True):
    '''
    return optimizer (name) in torch.optim.

    Args:
        net: model witth parameters to be optimized
        optim_name: optimizer name, SGD|AdamW
        lr: learning rate
        momentum: momentum parameter for SGD
        weight_decay: weight decay in optimizer
        layer_decay: layer-wise decay learning rate for model, requires the model have group_matcher function
        nesterov: SGD parameter
        bn_wd_skip: if bn_wd_skip, the optimizer does not apply weight decay regularization on parameters in batch normalization.
    '''
    assert layer_decay <= 1.0

    no_decay = {}
    if hasattr(net, 'no_weight_decay') and bn_wd_skip:
        no_decay = net.no_weight_decay()
    
    if layer_decay != 1.0:
        per_param_args = param_groups_layer_decay(net, lr, weight_decay, no_weight_decay_list=no_decay, layer_decay=layer_decay)
    else:
        per_param_args = param_groups_weight_decay(net, weight_decay, no_weight_decay_list=no_decay)

    if optim_name == 'SGD':
        optimizer = SGD(per_param_args, lr=lr, momentum=momentum, weight_decay=weight_decay,
                                    nesterov=nesterov)
    elif optim_name == 'AdamW':
        optimizer = AdamW(per_param_args, lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)

    return optimizer


def get_cosine_schedule_with_warmup(optimizer,
                                    num_training_steps,
                                    num_cycles=7. / 16.,
                                    num_warmup_steps=0,
                                    last_epoch=-1):
    '''
    Get cosine scheduler (LambdaLR).
    if warmup is needed, set num_warmup_steps (int) > 0.
    '''
    from torch.optim.lr_scheduler import LambdaLR
    def _lr_lambda(current_step):
        '''
        _lr_lambda returns a multiplicative factor given an interger parameter epochs.
        Decaying criteria: last_epoch
        '''

        if current_step < num_warmup_steps:
            _lr = float(current_step) / float(max(1, num_warmup_steps))
        else:
            num_cos_steps = float(current_step - num_warmup_steps)
            num_cos_steps = num_cos_steps / float(max(1, num_training_steps - num_warmup_steps))
            _lr = max(0.0, math.cos(math.pi * num_cycles * num_cos_steps))
        return _lr

    return LambdaLR(optimizer, _lr_lambda, last_epoch)

