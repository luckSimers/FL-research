import os
import copy
import torch
import logging
import numpy as np
import torchvision.datasets as datasets
from scipy.io import loadmat
from .dataset import BasicDataset
from .utils import split_labeled_unlabeled


def load_mnistm(base_dir, train=True):
    """
    numbers: 55000
    shape: 3x28x28
    class dist: [5444. 6179. 5470. 5638. 5307. 4987. 5417. 5715. 5389. 5454.]
    """
    mnistm_data = loadmat(base_dir + '/mnistm_with_label.mat')
    if train:
        data = mnistm_data['train']
        labels = mnistm_data['label_train']
    else:
        data = mnistm_data['test']
        labels = mnistm_data['label_test']
    data = data.transpose(0, 3, 1, 2)
    labels = labels.argmax(axis=1).astype(np.int64)
    np.place(labels, labels == 10, 0)
    logging.debug(f'loading mnistm done! train={train} shape={data.shape}')
    
    return data, labels

def load_syn(base_dir, train=True):
    """
    numbers: 25000
    shape: 3x32x32
    class dist: [2475. 2600. 2566. 2456. 2534. 2496. 2505. 2479. 2401. 2488.]
    """
    syn_data = loadmat(base_dir + '/syn_number.mat')
    if train:
        data = syn_data['train_data']
        labels = syn_data['train_label']
    else:
        data = syn_data['test_data']
        labels = syn_data['test_label']
    data = data.transpose(0, 3, 1, 2)
    labels = labels.astype(np.int64).squeeze()
    np.place(labels, labels == 10, 0)
    logging.debug(f'loading syn done! train={train} shape={data.shape}')
    return data, labels

def load_digits(data_root, train=True, lb_per_class=0, labeled_set='SVHN',):
    '''
    return a dict of datasets {
        'MNISTM': mnistm,
        'SYN': syn,
        'USPS': usps,
        'SVHN': svhn,
        'MNIST': mnist,
        'lb_set': lb_set
    }
    '''
    # load data from each dataset
    digits = {}
    num_sample = 7000
    classes = 10
    if not train:
        num_sample = None
    imgs_l = None
    # load mnistm
    imgs, labels = load_mnistm(os.path.join(data_root, 'DIGIT5'), train=train)    
    perm = np.random.permutation(imgs.shape[0])
    if labeled_set == 'MNISTM' and train and lb_per_class > 0:
        imgs_l, label_l =  imgs[perm][num_sample:], labels[perm][num_sample:]
        imgs_l, label_l, _, _ = split_labeled_unlabeled(imgs_l, label_l, 10, lb_per_class)
    imgs, labels = imgs[perm][:num_sample], labels[perm][:num_sample]
    digits['MNISTM'] = BasicDataset(
        'MNISTM', imgs, labels, classes, is_train=train)
    
    # load syn
    imgs, labels = load_syn(os.path.join(data_root, 'DIGIT5'), train=train)
    perm = np.random.permutation(imgs.shape[0])
    if labeled_set == 'SYN' and train and lb_per_class > 0:
        imgs_l, label_l = imgs[perm][num_sample:], labels[perm][num_sample:]
        imgs_l, label_l, _, _ = split_labeled_unlabeled(imgs_l, label_l, 10, lb_per_class)
    imgs, labels = imgs[perm][:num_sample], labels[perm][:num_sample]
    digits['SYN'] = BasicDataset(
        'SYN', imgs, labels, classes, is_train=train)
    
    # load usps
    dset = datasets.USPS(root=os.path.join(data_root, 'USPS'), train=train, download=True)
    imgs, labels = dset.data, np.array(dset.targets).astype(np.int64)
    perm = np.random.permutation(imgs.shape[0])
    if labeled_set == 'USPS' and train and lb_per_class > 0:
        imgs_l, label_l =  imgs[perm][num_sample:], labels[perm][num_sample:]
        imgs_l, label_l, _, _ = split_labeled_unlabeled(imgs_l, label_l, 10, lb_per_class)
    imgs, labels = imgs[perm][:num_sample], labels[perm][:num_sample]
    digits['USPS'] = BasicDataset(
        'USPS', imgs, labels, classes, is_train=train)
    
    # load svhn
    dset = datasets.SVHN(root=os.path.join(data_root, 'SVHN'), split='train' if train else 'test', download=True)
    imgs, labels = dset.data, dset.labels.astype(np.int64)
    perm = np.random.permutation(imgs.shape[0])
    if labeled_set == 'SVHN' and train and lb_per_class > 0:
        imgs_l, label_l =  imgs[perm][num_sample:], labels[perm][num_sample:]
        imgs_l, label_l, _, _ = split_labeled_unlabeled(imgs_l, label_l, 10, lb_per_class)
    imgs, labels = imgs[perm][:num_sample], labels[perm][:num_sample]
    digits['SVHN'] = BasicDataset(
        'SVHN', imgs, labels, classes, is_train=train)
    
    # load mnist
    dset = datasets.MNIST(root=os.path.join(data_root, 'MNIST'), train=train, download=True)
    imgs, labels = dset.data.numpy(), dset.targets.numpy().astype(np.int64)
    perm = np.random.permutation(imgs.shape[0])
    if labeled_set == 'MNIST' and train and lb_per_class > 0:
        imgs_l, label_l =  imgs[perm][num_sample:], labels[perm][num_sample:]
        imgs_l, label_l, _, _ = split_labeled_unlabeled(imgs_l, label_l, 10, lb_per_class)
    imgs, labels = imgs[perm][:num_sample], labels[perm][:num_sample]
    # imgs.resize((len(imgs), 1, 28, 28)) 
    digits['MNIST'] = BasicDataset(
        'MNIST', imgs, labels, classes, is_train=train)

    lb_set = None
    if imgs_l is not None:
        lb_set = BasicDataset(
            labeled_set, imgs_l, label_l, classes, is_train=train)
    digits['lb_set'] = lb_set        
    return digits

def load_office31(data_root, train=True, lb_per_class=0, labeled_set='SVHN'):
    num_sample = 1000
    raise NotImplementedError


def load_normal_dataset(data_root, dataset, train=True, lb_per_class=0):
    logging.info("load_normal_dataset {}...".format(dataset))
    dset = getattr(datasets, dataset)     #dataset.dataset类 用于加载数据
    data_root = os.path.join(data_root, dataset)
    if dataset == 'SVHN':
        if train:
            dset = dset(data_root, split='train', download=True)
        else:
            dset = dset(data_root, split='test', download=True)
        classes = 10
        imgs, labels = dset.data, dset.labels
    else: # CIFAR10, CIFAR100, MNIST
        dset = dset(data_root, train=train, download=True)
        classes = len(dset.classes)
        imgs, labels = dset.data, dset.targets
    
    if type(imgs) == torch.Tensor:
        imgs = imgs.numpy()
    if type(labels) == list:
        labels = np.array(labels)
    elif type(labels) == torch.Tensor:
        labels = labels.numpy()
    labels = labels.astype(np.int64)

    perm = np.random.permutation(imgs.shape[0])
    imgs, labels = imgs[perm], labels[perm]
    if train and lb_per_class > 0:
        img_lb, labels_lb, img_ulb, labels_ulb = split_labeled_unlabeled(imgs, labels, classes, lb_per_class)
        lb_set = BasicDataset(
            dataset, img_lb, labels_lb, classes, is_train=train)
        ulb_set = BasicDataset(
            dataset, img_ulb, labels_ulb, classes, is_train=train)
    else:
        lb_set = None
        ulb_set = BasicDataset(
            dataset, imgs, labels, classes, is_train=train)
    return {
        'lb_set': lb_set,
        dataset: ulb_set,
    }
    

def fetch_dataset(data_root, dataset, lb_per_class=0, labeled_set='', train=True):
    """
    return Dict :{
        subset: BasicDataset,
        ...,
        lb_set: BasicDataset
    }
    """
    if dataset in ['DIGIT5', 'Office31'] and lb_per_class > 0 and labeled_set == '':
        raise ValueError('labeled_set must be specified when lb_per_class > 0')
    
    logging.info(f'fetching dataset-{dataset} from {data_root}...')
    if dataset in ['MNIST', 'CIFAR10', 'CIFAR100', 'SVHN']:
        return load_normal_dataset(data_root, dataset, train, lb_per_class)
    
    elif dataset == 'DIGIT5':
        return load_digits(data_root, train, lb_per_class, labeled_set)
    
    elif dataset == 'Office31':
        return load_office31(data_root, train, lb_per_class, labeled_set)

    else:
        raise ValueError('Not valid dataset name')


        
    
