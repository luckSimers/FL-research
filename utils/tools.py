import os
import time
import torch
import random
import logging
import numpy as np
import torch.nn as nn
import torch.cuda as cuda
from torch.utils.data import DataLoader



def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cuda.manual_seed(seed)
    cuda.manual_seed_all(seed)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.max = -float("inf")
        self.min = float("inf")
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.max = val if val > self.max else self.max
        self.min = val if val < self.min else self.min

    def make_summary(self, key="None"):
        sum_key = key + "/" + "sum"
        count_key = key + "/" + "count"
        avg_key = key + "/" + "avg"
        max_key = key + "/" + "max"
        min_key = key + "/" + "min"
        final_key = key + "/" + "final"
        summary = {
            sum_key: self.sum,
            count_key: self.count,
            avg_key: self.avg,
            max_key: self.max,
            min_key: self.min,
            final_key: self.val,
        }
        return summary
    
def get_logger(name, save_path=None, level='INFO'):
    """
    create logger function
    """
    logger = logging.getLogger(name)
    logging.basicConfig(format='%(filename)s-[%(lineno)d]: %(message)s', level=getattr(logging, level.upper()))

    if not save_path is None:
        os.makedirs(save_path, exist_ok=True)
        log_format = logging.Formatter('%(filename)s-[%(lineno)d]: %(message)s')
        fileHandler = logging.FileHandler(os.path.join(save_path, f'log_{time.asctime().replace(" ", "_")}.txt'))
        fileHandler.setFormatter(log_format)
        logger.addHandler(fileHandler)

    return logger

def make_batchnorm(m, momentum, track_running_stats):
    """"
        change the set of batchNorm layer
    """
    if isinstance(m, nn.BatchNorm2d):
        m.momentum = momentum
        m.track_running_stats = track_running_stats
        if track_running_stats:
            m.register_buffer('running_mean', torch.zeros(m.num_features, device=m.weight.device))
            m.register_buffer('running_var', torch.ones(m.num_features, device=m.weight.device))
            m.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long, device=m.weight.device))
        else:
            m.register_buffer('running_mean', None)
            m.register_buffer('running_var', None)
            m.register_buffer('num_batches_tracked', None)

def make_batchnorm_stats(datasets, test_model, device):
    """
        update the batchnorm layer
    """
    with torch.no_grad():
        test_model.apply(lambda m: make_batchnorm(m, momentum=None, track_running_stats=True))
        if isinstance(datasets, dict):
            for name, dataset in datasets.items():
                data_loader = DataLoader(dataset, batch_size=500)
                test_model.train(True)
                for i, data in enumerate(data_loader):
                    x = data['x'].to(device)
                    test_model(x)
        else:
            data_loader = DataLoader(datasets, batch_size=500)
            test_model.train(True)
            for i, data in enumerate(data_loader):
                x = data['x'].to(device)
                test_model(x)

class Special_Argument(object):
    """
    Algrithm specific argument
    """
    def __init__(self, name, type, default, help=''):
        """
        Model specific arguments should be added via this class.
        """
        self.name = name
        self.type = type
        self.default = default
        self.help = help

def plot_tsne(model, dataloader, device, use_prototype=False):
    '''
    show the feature space in the 2d figure
    '''
    model.eval()
    all_reps = []
    all_y = []
    for data in dataloader:
        x, y = data['x_lb'].to(device), data['y_lb'].to(device)
        with torch.no_grad():
            reps = model.base(x)
            all_reps.append(reps)
            all_y.append(y)
    reps = torch.cat(all_reps, dim=0)
    y = torch.cat(all_y, dim=0)

    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    X_tsne = tsne.fit_transform(reps.cpu().numpy())
    plt.figure(figsize=(10, 10))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y.cpu().numpy(), s=10, cmap='Set1_r')
    plt.colorbar(ticks=range(3))
    plt.savefig(f'{use_prototype}_tsne.png')

def get_params(model, detach=True) -> torch.Tensor:
    params = None
    for p in model.parameters():
        if p.requires_grad:
            if detach:
                if params is None:
                    params = p.data.detach().view(-1)
                else:
                    params = torch.cat((params, p.data.detach().view(-1)), dim=0)
            else:
                if params is None:
                    params = p.data.view(-1)
                else:
                    params = torch.cat((params, p.data.view(-1)), dim=0)
    return params # type: ignore

def mixup_data(x, y, alpha=0.75):
    lam = np.random.beta(alpha, alpha)
    b = x.size()[0]
    index = torch.randperm(b)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

