import os
import math
import logging
import torch.nn as nn
import torch.optim as optim
import nets as models


def get_net_builder(name):
    '''
    return a function that builds a model from package net
    Args:
        name: model name
    '''
    model_dict = sorted(name for name in models.__dict__ 
                        if callable(models.__dict__[name]))
    if name in model_dict:
        return models.__dict__[name]
    else:
        raise Exception(f'No model named {name},\nexpected: {model_dict},\nreceived: {name}')
    
def param_groups_weight_decay(
        model: nn.Module,
        weight_decay=1e-5,
        no_weight_decay_list=()
):
    '''
    divide parameters into two part: one is need to decay,other is not
    '''
    # Ref: https://github.com/rwightman/pytorch-image-models
    no_weight_decay_list = set(no_weight_decay_list)
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.ndim <= 1 or name.endswith(".bias") or name in no_weight_decay_list:
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]

def param_groups_weight_decay_sigma(
        model: nn.Module,
        weight_decay=1e-5,
        no_weight_decay_list=()
):
    '''
    
    '''
    # Ref: https://github.com/rwightman/pytorch-image-models
    no_weight_decay_list = set(no_weight_decay_list)
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if 'phi' in name:
            continue

        if param.ndim <= 1 or name.endswith(".bias") or name in no_weight_decay_list:
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


def param_groups_weight_decay_phi(
        model: nn.Module,
        weight_decay=1e-5,
        no_weight_decay_list=()
):
    # Ref: https://github.com/rwightman/pytorch-image-models
    no_weight_decay_list = set(no_weight_decay_list)
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if 'sigma' in name:
            continue

        if param.ndim <= 1 or name.endswith(".bias") or name in no_weight_decay_list:
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]

def get_optimizer(net, optim_name='SGD', lr=0.1, momentum=0.9, weight_decay=0.0005, nesterov=True, bn_wd_skip=False, dec='full'):
    '''
    return optimizer (name) in optim.

    Args:
        net: model witth parameters to be optimized
        optim_name: optimizer name, SGD|AdamW
        lr: learning rate
        momentum: momentum parameter for SGD
        weight_decay: weight decay in optimizer
        layer_decay: layer-wise decay learning rate for model, requires the model have group_matcher function
        nesterov: SGD parameter
        bn_wd_skip: if bn_wd_skip, the optimizer does not apply weight decay regularization on parameters in batch normalization.
        dec: parameter decomposition, full|sigma|phi
    '''

    no_decay = {}
    if hasattr(net, 'no_weight_decay') and bn_wd_skip:
        no_decay = net.no_weight_decay()
    if dec == 'full':
        per_param_args = param_groups_weight_decay(net, weight_decay, no_weight_decay_list=no_decay)
    elif dec == 'sigma':
        per_param_args = param_groups_weight_decay_sigma(net, weight_decay, no_weight_decay_list=no_decay)
    else:
        per_param_args = param_groups_weight_decay_phi(net, weight_decay, no_weight_decay_list=no_decay)
    
    if optim_name == 'SGD':
        optimizer = optim.SGD(per_param_args, lr=lr, momentum=momentum, weight_decay=weight_decay,
                                    nesterov=nesterov)
    elif optim_name == 'AdamW':
        optimizer = optim.AdamW(per_param_args, lr=lr, weight_decay=weight_decay)

    else:
        raise NotImplementedError('Optimizer {} is not implemented.'.format(optim_name))
    
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

