import copy
import logging
import numpy as np 
from PIL import Image
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
from ..utils import create_ft_set, visualize


class Subset4FL(Dataset):

    def __init__(
            self, 
            data_name,
            dataset,
            dataidxs=None,
            transform=None,
            target_transform=None,
            freq=False,
            del_num=0,
            random=False,
    ):
        self.data_name = data_name
        self.dataidxs = dataidxs
        self.dataset = dataset
        self.freq = freq
        self.del_num = del_num
        self.random = random
        if transform is not None:
            self.transform = transform
        else:
            self.transform = self.dataset.transform
        if target_transform is not None:
            self.target_transform = target_transform
        else:
            self.target_transform = self.dataset.target_transform
        self.data, self.targets = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
       
        if self.dataidxs is None:
            return self.dataset.data, self.dataset.targets
        data = self.dataset.data[self.dataidxs]
        targets = np.array(self.dataset.targets)[self.dataidxs]
        if self.freq:
            compressed_data = create_ft_set(data, self.del_num, self.random)
            visualize(data, compressed_data, self.data_name)
            return compressed_data, targets
        return data, targets


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, targets) where targets is index of the targets class.
        """
        img, targets = self.data[index], self.targets[index]
        if isinstance(img, np.ndarray):
            if self.data_name == 'SVHN':
                img = Image.fromarray(np.transpose(img, (1, 2, 0)))
            else:
                img = Image.fromarray(img)
        
        if self.transform is not None and not self.freq:
            img = self.transform(img)

        if self.target_transform is not None:
            targets = self.target_transform(targets)

        return img, targets

    def __len__(self):
        return len(self.data)


# class FinetuneSet(Dataset):

#     def __init__(
#             self, 
#             data_name,
#             dataset,
#             data_per_cls,
#             transform=None,
#             target_transform=None,
#     ):
#         self.data_name = data_name
#         self.dataset = dataset
#         self.data_per_cls = data_per_cls
#         if transform is not None:
#             self.transform = transform
#         else:
#             self.transform = self.dataset.transform
#         if target_transform is not None:
#             self.target_transform = target_transform
#         else:
#             self.target_transform = self.dataset.target_transform
#         self.data, self.targets = self.__build_truncated_dataset__()

#     def __build_truncated_dataset__(self):
       
#         targets = np.array(self.dataset.targets)
#         idx = []
#         for i in range(len(self.dataset.classes)):
#             idx_i = np.where(targets == i)[0]
#             np.random.shuffle(idx_i)
#             idx_i = idx_i[:self.data_per_cls]
#             idx.extend(idx_i)
#         data = self.dataset.data[idx]
#         targets = targets[idx]
#         return data, targets


#     def __getitem__(self, index):
#         """
#         Args:
#             index (int): Index

#         Returns:
#             tuple: (image, targets) where targets is index of the targets class.
#         """
#         img, targets = self.data[index], self.targets[index]
#         if isinstance(img, np.ndarray):
#             if self.data_name == 'SVHN':
#                 img = Image.fromarray(np.transpose(img, (1, 2, 0)))
#             else:
#                 img = Image.fromarray(img)
        
#         if self.transform is not None:
#             img = self.transform(img)

#         if self.target_transform is not None:
#             targets = self.target_transform(targets)

#         return img, targets

#     def __len__(self):
#         return len(self.data)
