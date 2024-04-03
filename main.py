import argparse
import logging
import os, re
import time
import sys
import wandb
from utils import logging_config, set_random_seed

# sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
# sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

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

from algorithms import FedAvg
from algorithms import Centralized
from algorithms import FedProx, FedDyn, SCAFFOLD
name2algo = {
    'centralized': Centralized,
    'fedavg': FedAvg,
    'fedprox': FedProx,
    'feddyn': FedDyn,
    'scaffold': SCAFFOLD,

}


def get_config():
    
    parser = argparse.ArgumentParser()
    '''
    Saving & loading of the model.
    '''
    parser.add_argument('--pj_name', type=str, default='FL')
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--load_path', type=str, default='')
    '''
    '''
    
    parser.add_argument('--eval_gap', type=int, default=1, help='evaluation frequency')
    parser.add_argument('--log_gap', type=int, default=1, help='logging frequency')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--c_batch_size', type=int, default=128)
    parser.add_argument('--eval_bs', type=int, default=256,
                        help='batch size of evaluation data loader (it does not affect the accuracy)')
    '''
    Optimizer configurations
    '''
    parser.add_argument('--sBN', type=bool, default=False, help='use sBN or not')
    parser.add_argument('--optim', type=str, default='SGD')
    parser.add_argument('--lr', type=float, default=3e-2)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--clip_grad', type=float, default=1.0, help='clip_grad')
    '''
    Backbone Net Configurations
    '''
    parser.add_argument('--net', type=str, default='ModerateCNN')

    '''
    Algorithms Configurations
    '''  
    ## core algorithm setting
    parser.add_argument('--algorithm', type=str, default='fedavg', help='')
    parser.add_argument('--ft_data_per_cls', type=int, default=0, help='data per class for server finetuning')
    parser.add_argument('--compress_freq', type=bool, default=False, help='compress fine-tune set in frequency domain')
    parser.add_argument('--del_freq_chs', type=int, default=0, help='number of high-frequency channels to be deleted')
    parser.add_argument('--randomize', type=bool, default=False, help='whether randomize left high-frequency channels')

    
    '''
    Data Configurations
    '''
    ## standard setting configurations
    parser.add_argument('--data_dir', type=str, default='/remote-home/hongquanliu/Datasets')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--num_workers', type=int, default=1)

    ## Federated Learning setting configurations
    parser.add_argument('--global_rounds', type=int, default=100)
    parser.add_argument('--local_steps', type=int, default=5, help='number of local steps')
    parser.add_argument('--client_num', type=int, default=10)
    parser.add_argument('--join_ratio', type=float, default=1, help='ratio of clients to join in each round')
    parser.add_argument('--split_type', type=str, default='iid', help='type of heterogeneity')

    # system config：
    parser.add_argument('--level', type=str, default='info', help='logging level',)
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--visible_gpu', type=str, default='0')
  
    # config file
    parser.add_argument('--c', type=str, default='')

    args = parser.parse_args()
    over_write_args_from_file(args, args.c)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.visible_gpu

    for argument in name2algo[args.algorithm].get_argument():
        parser.add_argument(argument.name, type=argument.type, default=argument.default, help=argument.help)

    args = parser.parse_args()
    over_write_args_from_file(args, args.c)

    args.exp_tag = f'{args.dataset}'
    if args.algorithm == 'centralized':
        args.exp_tag += f'_{args.algorithm}'
    else:
        args.exp_tag += f'_{args.split_type}_{args.client_num}_{args.algorithm}_ft-{args.ft_data_per_cls}'

    for argument in name2algo[args.algorithm].get_argument():
        name = argument.name[2:]
        args.exp_tag += f'_{name}={getattr(args, name)}'
    args.exp_tag += f'_seed={args.seed}'
    args.num_gpu = len(re.findall(r'\d+', args.visible_gpu))
    args.dataset = args.dataset.upper()
    return args

def init_wandb(args):
    print(f'-----------------init_wandb:{args.exp_tag}-----------------')
    project = args.pj_name
    name = args.exp_tag # key hyperparams

    # tags
    alg = f'alg={args.algorithm}'
    bs = f'bs={args.batch_size}'
    ds = f'ds={args.dataset}'
    net = f'net={args.net}'
    if args.algorithm == 'centralized':
        tags = [alg, bs, ds, net]
    else:
        st = f'st={args.split_type}'
        nc = f'nc={args.client_num}'
        ft = f'ft={args.ft_data_per_cls}'
        tags = [ alg, bs, ds, st, nc, net]

    run = wandb.init(name=name, 
                    tags=tags, 
                    config=args.__dict__, 
                    project=project,
                    mode="offline",
            )
    return run

os.environ["WANDB__SERVICE_WAIT"] = '300'
def main():
    cfg = get_config()
    set_random_seed(cfg.seed)
    logging_config(args=cfg)
    cfg.logger = init_wandb(cfg)
    algorithm = name2algo[cfg.algorithm](cfg)
    start = time.time()
    algorithm.run()
    end = time.time()
    logging.info(f'time cost:{(end-start)/3600:.2f}h')


if __name__ == '__main__':
    main()    




