import logging
import random
import math
import functools
import os
import json
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets



data_stats = {'MNIST': ((0.1307,), (0.3081,)), 'FashionMNIST': ((0.2860,), (0.3530,)),
              'CIFAR10': ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
              'CIFAR100': ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
              'SVHN': ((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
              'STL10': ((0.4409, 0.4279, 0.3868), (0.2683, 0.2610, 0.2687))}


def record_net_data_stats(y_train, net_dataidx):
    client_train_cls_counts_dict = {}

    for client_idx, dataidx in enumerate(net_dataidx):
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True) 
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        client_train_cls_counts_dict[client_idx] = tmp
    logging.info('Data statistics: %s' % str(client_train_cls_counts_dict))
    return client_train_cls_counts_dict

class FetchData(object):

    image_resolution_dict = {
        "cifar10": 32,
        "cifar100": 32,
        "SVHN": 32,
        "fmnist": 32,
    }

    def __init__(
            self, 
            dataset="", 
            datadir="./",
            partition_type="iid", 
            client_number=1, 
            ft_data=0,
            num_workers=4, 
            ):

        # less use this.
        # For partition
        self.dataset = dataset.upper()
        self.datadir = datadir
        self.partition_type = partition_type
        self.client_number = client_number
        self.num_workers = num_workers
        self.ft_data = ft_data
        self.other_params = {}


    def load_data(self):
        if self.client_number > 1:
            train_ds, test_ds = self.federated_split() 
            self.other_params["local_counts"] = self.local_counts
            self.other_params["client_idx"] = self.client_idx
            return train_ds, test_ds, self.local_data_num, self.class_num, self.other_params
        train_ds, test_ds = self.fetch_dataset()
        self.other_params["local_counts"] = None
        self.other_params["client_idx"] = None
        return train_ds, test_ds, len(train_ds), self.class_num, self.other_params

    def fetch_dataset(self):
        '''
        load dataset: data_name
        return dataset{
            'trainset': train set,
            'testset': test set
        }
        '''
        data_name = self.dataset
        data_dir = self.datadir
        logging.debug('fetching data of {}-dataset'.format(data_name))
        data_dir = os.path.join(data_dir, data_name)
        dset = getattr(datasets, data_name)
        if data_name in ['MNIST', 'FashionMNIST', 'CIFAR10', 'CIFAR100', 'SVHN']:
            if data_name in ['MNIST', 'FashionMNIST']:
                train_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(*data_stats[data_name])])
                test_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(*data_stats[data_name])])
                trainset = dset(data_dir, train=True, download=True, transform=train_transform)
                testset = dset(data_dir, train=False, download=True, transform=test_transform)
            elif data_name in ['CIFAR10', 'CIFAR100']:
                train_transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                    transforms.ToTensor(),
                    transforms.Normalize(*data_stats[data_name])])
                test_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(*data_stats[data_name])])
                trainset = dset(data_dir, train=True, download=True, transform=train_transform)
                testset = dset(data_dir, train=False, download=True, transform=test_transform)
            else:
                train_transform = transforms.Compose([
                    transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                    transforms.ToTensor(),
                    transforms.Normalize(*data_stats[data_name])])
                test_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(*data_stats[data_name])])
                trainset = dset(data_dir, split='train', download=True, transform=train_transform)
                testset = dset(data_dir, split='test', download=True, transform=test_transform)
                # SVHN training split comprises 73257 digits, testing split comprises 26032 digits.

            if not hasattr(trainset, 'targets'):
                ## align SVHN with CIFAR dataset
                trainset.targets = trainset.labels
                testset.targets = testset.labels
            
            if not hasattr(trainset, 'classes'):
                ## align SVHN with CIFAR dataset
                class_num =  len(set(testset.targets))
                trainset.classes = list(range(class_num))
                testset.classes = list(range(class_num))

            self.class_num = len(trainset.classes)

            logging.info('default augmentation for train set: {}'.format(train_transform))
            logging.info('dataset: {}, classes: {}, {} train samples and {} test samples'.format(
                data_name, self.class_num, len(trainset), len(testset)))
        else:
            raise NotImplementedError
        return trainset, testset


    def get_y_train_np(self, train_ds):
        if self.dataset in ["fmnist"]:
            y_train = train_ds.targets.data
        else:
            y_train = train_ds.targets
        y_train_np = np.array(y_train)
        return y_train_np


    def federated_split(self):
        logging.debug("federated_split")
        train_ds, test_ds = self.fetch_dataset()
        self.other_params['ft_idx'] = None
        if self.ft_data > 0:
            targets = np.array(train_ds.targets)
            idx = []
            for i in range(len(train_ds.classes)):
                idx_i = np.where(targets == i)[0]
                np.random.shuffle(idx_i)
                idx_i = idx_i[:self.ft_data]
                idx.extend(idx_i)
            self.other_params['ft_idx'] = idx

        y_train_np = self.get_y_train_np(train_ds)  
        self.global_train_num = y_train_np.shape[0]
        self.global_test_num = len(test_ds) 

        self.client_idx, self.local_counts = self.partition_data(y_train_np, self.global_train_num)

        self.local_data_num = [] 
        # self.local_train_loader = []
        for client_index in range(self.client_number):
            # train_ds= self.load_sub_data(client_index, train_ds, test_ds)
            self.local_data_num.append(len(self.client_idx[client_index]))
            # train_dl = data.DataLoader(dataset=train_ds, batch_size=self.batch_size, shuffle=True, 
            #                     drop_last=True, num_workers=self.num_workers)
            # self.local_train_loader.append(train_dl)
        return train_ds, test_ds


    def partition_data(self, y_train_np, train_data_num):

        logging.info("partition_type = " + (self.partition_type))
        load_path = os.path.join(self.datadir, self.dataset, "partition", self.partition_type + "_" + str(self.client_number) + '.npy')
        if os.path.exists(load_path):
            logging.debug("directly loading existing partition from: " + load_path)
            client_idx = np.load(load_path, allow_pickle=True)
            local_counts = record_net_data_stats(y_train_np, client_idx)
            return client_idx, local_counts
        

        if self.partition_type == "iid":
            logging.debug('IID partitioning')
            total_num = train_data_num
            idxs = np.random.permutation(total_num)
            batch_idxs = np.array_split(idxs, self.client_number)
            client_idx = {i: batch_idxs[i] for i in range(self.client_number)}

        elif self.partition_type.startswith('dir'):
            logging.debug('dirichlet Partitioning')
            alpha = float(self.partition_type.split('_')[1])
            min_size = 0
            K = self.class_num    
            N = y_train_np.shape[0] 
            logging.info("N = " + str(N))
            client_idx = {}  
            while min_size < K:
                idx_batch = [[] for _ in range(self.client_number)]
              
                for k in range(K): 
                    idx_k = np.where(y_train_np == k)[0]
                    np.random.shuffle(idx_k)
                    proportions = np.random.dirichlet(np.repeat(alpha, self.client_number))
                    proportions = np.array([p * (len(idx_j) < N / self.client_number) for p, idx_j in zip(proportions, idx_batch)])
                    proportions = proportions / proportions.sum()
                    proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                    idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                    min_size = min([len(idx_j) for idx_j in idx_batch])

            for j in range(self.client_number):
                np.random.shuffle(idx_batch[j])
                client_idx[j] = idx_batch[j]

        elif self.partition_type.startswith('path'):
            logging.debug('Pathological partitioning')
            shard_per_user = int(self.partition_type.split('_')[1])
            client_idx = {}  
            shard_per_class = int(shard_per_user * self.client_number / self.class_num)
            target_idx_split = {}
            for target_i in range(self.class_num):
                target_idx = np.where(y_train_np == target_i)[0]
                num_leftover = len(target_idx) % shard_per_class
                leftover = target_idx[-num_leftover:] if num_leftover > 0 else []
                new_target_idx = target_idx[:-num_leftover] if num_leftover > 0 else target_idx
                new_target_idx = np.split(new_target_idx, shard_per_class)
                for i, leftover_target_idx in enumerate(leftover):
                    new_target_idx[i] = np.append(new_target_idx[i], leftover_target_idx)
                target_idx_split[target_i] = new_target_idx
            target_split = np.array(list(range(self.class_num)) * shard_per_class)
            target_split = np.random.permutation(target_split).reshape((self.client_number, -1))
            for i in range(self.client_number):
                for target_i in target_split[i]:
                    idx = np.random.randint(len(target_idx_split[target_i]))
                    client_idx[i] = np.append(client_idx[i], target_idx_split[target_i][idx])
                    target_idx_split[target_i] = np.delete(target_idx_split[target_i], idx, axis=0)


        client_idx = [client_idx[i] for i in range(self.client_number)]
        os.makedirs(os.path.dirname(load_path), exist_ok=True)
        np.save(load_path, client_idx)
        logging.debug("partition saved to: " + load_path)
        local_counts = record_net_data_stats(y_train_np, client_idx)

        return client_idx, local_counts




