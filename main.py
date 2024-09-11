import os
import sys
import time
import wandb
import torch
import argparse
from utils import get_logger, set_random_seed

from algorithms import FedAvg
from algorithms import Centralized
from algorithms import FedProx
name2algo = {
    'centralized': Centralized,
    'fedavg': FedAvg,
    'fedprox': FedProx,

}

shape = {
    'CIFAR10': [3, 32, 32],
    'CIFAR100': [3, 32, 32],
    'SVHN': [3, 32, 32],
}
classes = {
    'CIFAR10': 10,
    'CIFAR100': 100,
    'SVHN': 10,
}

def over_write_args_from_file(args, yml):
    """
    overwrite arguments acocrding to config file
    """
    import ruamel.yaml as yaml
    if yml == '':
        return
    with open(yml, 'r', encoding='utf-8') as f:
        dic = yaml.load(f.read(), Loader=yaml.Loader)
        for k in dic:
            setattr(args, k, dic[k])

def str2bool(v):
    """
    str to bool
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_config():
    
    parser = argparse.ArgumentParser()
    '''
    Saving & loading of the model.
    '''
    parser.add_argument('--pj_name', type=str, default='FL')
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--load_path', type=str, default='')
    parser.add_argument('--eval_gap', type=int, default=1, help='evaluation frequency')
    parser.add_argument('--log_gap', type=int, default=1, help='logging frequency')
    parser.add_argument('--bs', type=int, default=20)
    parser.add_argument('--c_bs', type=int, default=64)
    parser.add_argument('--eval_bs', type=int, default=256,
                        help='batch size of evaluation data loader (it does not affect the accuracy)')
    '''
    Optimizer configurations
    '''
    parser.add_argument('--sBN', type=str2bool, default=False, help='use sBN or not')
    parser.add_argument('--optim', type=str, default='SGD')
    parser.add_argument('--lr', type=float, default=3e-2)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--clip_grad', type=float, default=1.0, help='clip_grad')
    '''
    Backbone Net Configurations
    '''
    parser.add_argument('--net', type=str, default='resnet18')

    '''
    Algorithms Configurations
    '''  
    ## core algorithm setting
    parser.add_argument('--algorithm', type=str, default='fedavg', help='')
    parser.add_argument('--ft_data_per_cls', type=int, default=0, help='data per class for server finetuning')
    parser.add_argument('--compress_freq', type=str2bool, default=False, help='compress fine-tune set in frequency domain')
    parser.add_argument('--del_freq_chs', type=int, default=0, help='number of high-frequency channels to be deleted')
    parser.add_argument('--randomize', type=str2bool, default=False, help='whether randomize left high-frequency channels')
    
    '''
    Data Configurations
    '''
    ## standard setting configurations
    parser.add_argument('--data_dir', type=str, default='/remote-home/share/fl_dataset')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--num_workers', type=int, default=1)

    ## Federated Learning setting configurations
    parser.add_argument('--global_rounds', type=int, default=800)
    parser.add_argument('--c_steps', type=int, default=5, help='number of local steps')
    parser.add_argument('--s_steps', type=int, default=5, help='number of server local steps')
    parser.add_argument('--client_num', type=int, default=10)
    parser.add_argument('--join_ratio', type=float, default=0.5, help='ratio of clients to join in each round')
    parser.add_argument('--split_type', type=str, default='iid', help='iid, dir_x, pat_x')

    # system config：
    parser.add_argument('--level', type=str, default='info', help='logging level',)
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--visible_gpu', type=str, default='0')
  
    # FedProx Config
    parser.add_argument('--mu', type=float, default=0.1, help='proximal term')
    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.visible_gpu
    args.device = torch.device('cuda:0')

    if args.client_num == 10:
        args.join_ratio = 0.5
        args.c_bs = 64
    elif args.client_num == 100:
        args.join_ratio = 0.1
        args.c_bs = 32
    
    args.dataset = args.dataset.upper()
    args.data_shape = shape[args.dataset]
    args.num_classes = classes[args.dataset]

    if args.dataset in ['cifar10', 'cifar100', 'svhn']:
        args.img_sz = 32

    args.exp_tag = f'{args.algorithm}_{args.dataset}'
    if args.algorithm != 'centralized':
        args.exp_tag += f'_{args.split_type}_{args.client_num}'
        if args.sBN:
            args.exp_tag += '_sBN'
        if args.ft_data_per_cls:
            args.exp_tag += f'_ft={args.ft_data_per_cls}'
    args.exp_tag += f'_seed={args.seed}'
    
    return args

def init_wandb(args):
    print(f'-----------------init_wandb:{args.exp_tag}-----------------')
    os.environ["WANDB__SERVICE_WAIT"] = '300'
    project = args.pj_name
    name = args.exp_tag 
    run = wandb.init(name=name, 
                    config=args.__dict__, 
                    project=project,
                    mode="offline",
            )
    return run


def main():
    cfg = get_config()
    set_random_seed(cfg.seed)
    cfg.printer = get_logger(__name__, save_path=os.path.join(cfg.save_dir, cfg.exp_tag, 'logs'), level=cfg.level)
    cfg.logger = init_wandb(cfg)
    cfg.printer.info('configurations:')
    argv = sys.argv[1:]
    for i in range(len(sys.argv) // 2):
        cfg.printer.info(f'{argv[2*i][2:]}: {argv[2*i+1]}')
    cfg.printer.info('-' * 50)
    algorithm = name2algo[cfg.algorithm](cfg)
    start = time.time()
    algorithm.run()
    end = time.time()
    cfg.printer.info(f'time cost:{(end-start)/3600:.2f}h')


if __name__ == '__main__':
    main()    




