import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from .augmentation import RandAugment


data_stats = {
    'MNIST': ((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)), 
    'FashionMNIST': ((0.2860, 0.2860, 0.2860), (0.3530, 0.3530, 0.3530)),
    'CIFAR10': ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    'CIFAR100': ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    'SVHN': ((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
    'STL10': ((0.4409, 0.4279, 0.3868), (0.2683, 0.2610, 0.2687)),
    'MNISTM': ((0.0156, 0.0108, 0.0108), (0.1241, 0.1035, 0.1038)),
    'USPS': ((0.247, 0.243, 0.261), (0.292, 0.291, 0.297)),
    'SYN': ((0.0229, 0.0222, 0.0223), (0.1497, 0.1474, 0.1478)),
}

class BasicDataset(Dataset):
    """
    BasicDataset returns a pair of image and labels (targets).
    If targets are not given, BasicDataset returns None as the label.
    This class supports strong augmentation for FixMatch,
    and return both weakly and strongly augmented images.
    """

    def __init__(self,
                 data_name,
                 imgs, labels,
                 classes,
                 is_train=True,):
        """
        Args
            data: x_data
            targets: y_data (if not exist, None)
            num_classes: number of label classes
            transform: basic transformation of data
            use_strong_transform: If True, this dataset returns both weakly and strongly augmented images.
            strong_transform: list of transformation functions for strong augmentation
            onehot: If True, label is converted into onehot vector.
        """
        super(BasicDataset, self).__init__()

        self.data = imgs
        self.targets = labels
        self.classes = classes
        self.is_train = is_train
        self.pseudo_labels = None
        self.data_name = data_name

        if not is_train:
            trans = [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(*data_stats[data_name])
            ]
            self.transform = transforms.Compose(trans)
        
        else:
            trans = [
                transforms.Resize((32, 32)),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(*data_stats[data_name])
            ]
            strong_trans = [
                transforms.Resize((32, 32)),
                transforms.RandomCrop(32, padding=4),
                RandAugment(n=2, m=10),
                transforms.ToTensor(),
                transforms.Normalize(*data_stats[data_name])
            ]
            self.transform = transforms.Compose(trans)
            self.strong = transforms.Compose(strong_trans)

    def __sample__(self, idx):
        """ dataset specific sample function """
        # set idx-th target
        if self.targets is None:
            target = None
        else:
            target = self.targets[idx]

        if self.pseudo_labels is None:
            pseudo_labels = None
        else:
            pseudo_labels = self.pseudo_labels[idx]
            
        # set augmented images
        img = self.data[idx]
        return img, target, pseudo_labels

    def __getitem__(self, idx):
        """
        If strong augmentation is not used,
            return weak_augment_image, target
        else:
            return weak_augment_image, strong_augment_image, target
        """
        img, target, pseudo_label = self.__sample__(idx)

        if self.data_name in ['MNIST', 'USPS', 'FashionMNIST']:
            img = np.stack([img] * 3, axis=0)
        if self.data_name in ['CIFAR10', 'CIFAR100']:
            img = Image.fromarray(img)
        else: 
            img = Image.fromarray(np.transpose(img, (1,2,0)))
            

        if not self.is_train:
            img = self.transform(img)
            return {'idx': idx, 'x': img, 'y': target}
        elif pseudo_label is not None:
            return {'idx': idx, 'x': self.transform(img), 'x_s': self.strong(img), 'y': target, 'py': pseudo_label}   
        else:
            return {'idx': idx, 'x': self.transform(img), 'x_s': self.strong(img), 'y': target}   

    def __len__(self):
        return len(self.data)

class MixDataset(Dataset):
    def __init__(self, size, dataset):
        self.size = size
        self.dataset = dataset

    def __getitem__(self, index):
        index = torch.randint(0, len(self.dataset), (1,)).item()
        input = self.dataset[index]
        
        return input

    def __len__(self):
        return self.size

class AuxDataset(Dataset):
    def __init__(self, input, target):
        self.data = input
        self.targets = target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.targets[idx]
        return {'idx': idx, 'x': x, 'y': y}


class SubDataset(BasicDataset):
    def __init__(self, dataset, idx):
        super(SubDataset, self).__init__(dataset.data_name, dataset.data, dataset.targets, dataset.classes, dataset.is_train)
        self.data = self.data[idx]
        self.targets = self.targets[idx]