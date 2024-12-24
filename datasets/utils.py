import os
import copy
import torch
import random
import numpy as np
from io import BytesIO

# TODO: better way
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


def split_ssl_data(args, data, targets, num_classes,
                   lb_num_labels, ulb_num_labels=None,
                   lb_imbalance_ratio=1.0, ulb_imbalance_ratio=1.0,
                   lb_index=None, ulb_index=None, include_lb_to_ulb=True, load_exist=True):
    """
    data & target is splitted into labeled and unlabeled data.
    
    Args
        data: data to be split to labeled and unlabeled 
        targets: targets to be split to labeled and unlabeled 
        num_classes: number of total classes
        lb_num_labels: number of labeled samples. 
                       If lb_imbalance_ratio is 1.0, lb_num_labels denotes total number of samples.
                       Otherwise it denotes the number of samples in head class.
        ulb_num_labels: similar to lb_num_labels but for unlabeled data.
                        default to None, denoting use all remaining data except for labeled data as unlabeled set
        lb_imbalance_ratio: imbalance ratio for labeled data
        ulb_imbalance_ratio: imbalance ratio for unlabeled data
        lb_index: If np.array of index is given, select the data[index], target[index] as labeled samples.
        ulb_index: If np.array of index is given, select the data[index], target[index] as labeled samples.
        include_lb_to_ulb: If True, labeled data is also included in unlabeled data
    """
    data, targets = np.array(data), np.array(targets)
    lb_idx, ulb_idx = sample_labeled_unlabeled_data(args, data, targets, num_classes, 
                                                    lb_num_labels, ulb_num_labels,
                                                    lb_imbalance_ratio, ulb_imbalance_ratio, load_exist=False)
    
    # manually set lb_idx and ulb_idx, do not use except for debug
    if lb_index is not None:
        lb_idx = lb_index
    if ulb_index is not None:
        ulb_idx = ulb_index

    if include_lb_to_ulb:
        ulb_idx = np.concatenate([lb_idx, ulb_idx], axis=0)
    
    return data[lb_idx], targets[lb_idx], data[ulb_idx], targets[ulb_idx]



def sample_labeled_unlabeled_data(args, data, target, num_classes,
                                  lb_num_labels, ulb_num_labels=None,
                                  lb_imbalance_ratio=1.0, ulb_imbalance_ratio=1.0,
                                  load_exist=True):
    '''
    samples for labeled data
    (sampling with balanced ratio over classes)
    '''
    dump_dir = os.path.join(base_dir, 'data', args.dataset, 'labeled_idx')
    os.makedirs(dump_dir, exist_ok=True)
    lb_dump_path = os.path.join(dump_dir, f'lb_labels{args.num_labels}_{args.lb_imb_ratio}_seed{args.seed}_idx.npy')
    ulb_dump_path = os.path.join(dump_dir, f'ulb_labels{args.num_labels}_{args.ulb_imb_ratio}_seed{args.seed}_idx.npy')

    if os.path.exists(lb_dump_path) and os.path.exists(ulb_dump_path) and load_exist:
        lb_idx = np.load(lb_dump_path)
        ulb_idx = np.load(ulb_dump_path)
        return lb_idx, ulb_idx 

    
    # get samples per class
    if lb_imbalance_ratio == 1.0:
        # balanced setting, lb_num_labels is total number of labels for labeled data
        assert lb_num_labels % num_classes == 0, "lb_num_labels must be dividable by num_classes in balanced setting"
        lb_samples_per_class = [int(lb_num_labels / num_classes)] * num_classes
    else:
        # imbalanced setting, lb_num_labels is the maximum number of labels for class 1
        lb_samples_per_class = make_imbalance_data(lb_num_labels, num_classes, lb_imbalance_ratio)


    if ulb_imbalance_ratio == 1.0:
        # balanced setting
        if ulb_num_labels is None or ulb_num_labels == 'None':
            pass # ulb_samples_per_class = [int(len(data) / num_classes) - lb_samples_per_class[c] for c in range(num_classes)] # [int(len(data) / num_classes) - int(lb_num_labels / num_classes)] * num_classes
        else:
            assert ulb_num_labels % num_classes == 0, "ulb_num_labels must be dividable by num_classes in balanced setting"
            ulb_samples_per_class = [int(ulb_num_labels / num_classes)] * num_classes
    else:
        # imbalanced setting
        assert ulb_num_labels is not None, "ulb_num_labels must be set set in imbalanced setting"
        ulb_samples_per_class = make_imbalance_data(ulb_num_labels, num_classes, ulb_imbalance_ratio)

    lb_idx = []
    ulb_idx = []
    
    for c in range(num_classes):
        idx = np.where(target == c)[0]
        np.random.shuffle(idx)
        lb_idx.extend(idx[:lb_samples_per_class[c]])
        if ulb_num_labels is None or ulb_num_labels == 'None':
            ulb_idx.extend(idx[lb_samples_per_class[c]:])
        else:
            ulb_idx.extend(idx[lb_samples_per_class[c]:lb_samples_per_class[c]+ulb_samples_per_class[c]])
    
    if isinstance(lb_idx, list):
        lb_idx = np.asarray(lb_idx)
    if isinstance(ulb_idx, list):
        ulb_idx = np.asarray(ulb_idx)

    np.save(lb_dump_path, lb_idx)
    np.save(ulb_dump_path, ulb_idx)
    
    return lb_idx, ulb_idx


def make_imbalance_data(max_num_labels, num_classes, gamma):
    """
    calculate samplers per class for imbalanced data
    """
    mu = np.power(1 / abs(gamma), 1 / (num_classes - 1))
    samples_per_class = []
    for c in range(num_classes):
        if c == (num_classes - 1):
            samples_per_class.append(int(max_num_labels / abs(gamma)))
        else:
            samples_per_class.append(int(max_num_labels * np.power(mu, c)))
    if gamma < 0:
        samples_per_class = samples_per_class[::-1]
    return samples_per_class



def get_onehot(num_classes, idx):
    onehot = np.zeros([num_classes], dtype=np.float32)
    onehot[idx] += 1.0
    return onehot


def bytes_to_array(b: bytes) -> np.ndarray:
    np_bytes = BytesIO(b)
    return np.load(np_bytes, allow_pickle=True)


def random_subsample(wav: np.ndarray, max_length: float, sample_rate: int = 16000):
    """Randomly sample chunks of `max_length` seconds from the input audio"""
    sample_length = int(round(sample_rate * max_length))
    if len(wav) <= sample_length:
        return wav
    random_offset = random.randint(0, len(wav) - sample_length - 1)
    return wav[random_offset : random_offset + sample_length]


def split_labeled_unlabeled(imgs, labels, num_classes, lb_per_class):
    labeled_set = []
    for i in range(num_classes):
        idx = np.where(labels == i)[0]
        labeled_set.extend(idx[:lb_per_class])
    idx = list(range(len(labels)))
    unlabeled_idx = list(set(idx) - set(labeled_set))
    return imgs[labeled_set], labels[labeled_set], imgs[unlabeled_idx], labels[unlabeled_idx]

def split_dataset(dataset, args, num_clients):
    '''
    splitting {dataset} into {num_users} parts in IID or Non-IID mode
    return:
        data_split: [[data_idx_0], [data_idx_1], ...]
    '''
    split_type = args.split_type
    if split_type == 'iid':
        data_idx = iid(dataset, num_clients)
    elif split_type.split('_')[0] in ['pat', 'dir']:
        data_idx = non_iid(dataset, num_clients, split_type)
    else:
        raise ValueError('Not valid data split mode')
    return data_idx


def iid(dataset, num_users):
    '''
    splitting dataset into num_users parts for FL in IID
    '''
    num_items = int(len(dataset) / num_users)
    
    data_split = [[] for _ in range(num_users)]
    random_idx = torch.randperm(len(dataset))
    for i in range(num_users):
        if i == num_users - 1:
            data_split[i] = random_idx[i * num_items:].tolist()
        else:
            data_split[i] = random_idx[i * num_items: (i + 1) * num_items].tolist()
    return data_split


def non_iid(dataset, num_users, data_split_mode):
    num_classes = dataset.classes
    target = torch.tensor(dataset.targets)
    split_type, split_param = data_split_mode.split('_')
    data_split = [[] for i in range(num_users)]
    if split_type == 'pat':
        shard_per_user = int(split_param)
        target_idx_split = {}
        shard_per_class = int(shard_per_user * num_users / num_classes)
        if shard_per_class * num_classes != shard_per_user * num_users:
            raise ValueError('Not valid data split mode')
        for target_i in range(num_classes):
            target_idx = torch.where(target == target_i)[0]
            num_leftover = len(target_idx) % shard_per_class
            leftover = target_idx[-num_leftover:] if num_leftover > 0 else []
            new_target_idx = target_idx[:-num_leftover] if num_leftover > 0 else target_idx
            new_target_idx = new_target_idx.reshape((shard_per_class, -1)).tolist()
            for i, leftover_target_idx in enumerate(leftover):
                new_target_idx[i] = new_target_idx[i] + [leftover_target_idx.item()]
            target_idx_split[target_i] = new_target_idx
        target_split = list(range(num_classes)) * shard_per_class
        target_split = torch.tensor(target_split)[torch.randperm(len(target_split))].tolist()
        target_split = torch.tensor(target_split).reshape((num_users, -1)).tolist()
        for i in range(num_users):
            for target_i in target_split[i]:
                idx = torch.randint(len(target_idx_split[target_i]), (1,)).item()
                data_split[i].extend(target_idx_split[target_i].pop(idx))

    elif split_type == 'dir':
        beta = float(split_param)
        dir = torch.distributions.dirichlet.Dirichlet(torch.tensor(beta).repeat(num_users)) # type: ignore
        min_size = 0
        required_min_size = 10
        N = target.size(0)
        while min_size < required_min_size:
            data_split = [[] for _ in range(num_users)]
            for target_i in range(num_classes):
                target_idx = torch.where(target == target_i)[0]
                proportions = dir.sample()
                proportions = torch.tensor(
                    [p * (len(data_split_idx) < (N / num_users)) for p, data_split_idx in zip(proportions, data_split)])
                proportions = proportions / proportions.sum()
                split_idx = (torch.cumsum(proportions, dim=-1) * len(target_idx)).long().tolist()[:-1]
                split_idx = torch.tensor_split(target_idx, split_idx)
                data_split = [data_split_idx + idx.tolist() for data_split_idx, idx in zip(data_split, split_idx)]
            min_size = min([len(data_split_idx) for data_split_idx in data_split])
    else:
        raise ValueError('Not valid data split mode tag')
    return data_split



def separate_dataset(dataset, idx):
    separated_dataset = copy.deepcopy(dataset)
    separated_dataset.data = [dataset.data[s] for s in idx]
    separated_dataset.targets = [dataset.targets[s] for s in idx]
    return separated_dataset


def make_batchnorm_dataset_su(server_dataset, client_dataset):
    batchnorm_dataset = copy.deepcopy(server_dataset)
    batchnorm_dataset.data = batchnorm_dataset.data + client_dataset.data
    batchnorm_dataset.target = batchnorm_dataset.target + client_dataset.target
    batchnorm_dataset.other['id'] = batchnorm_dataset.other['id'] + client_dataset.other['id']
    return batchnorm_dataset
