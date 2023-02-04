import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms.functional as F


class NominalCIFAR10ImageDataset(Dataset):
    def __init__(self, nominal_class, transform, train=True):
        self.data = torchvision.datasets.CIFAR10(
            'datasets',
            train=train,
            download=True,
            transform=transform
        )

        self.indices = torch.where(torch.as_tensor(self.data.targets) == nominal_class)[0]

    def __getitem__(self, item):
        return self.data[self.indices[item % len(self.indices)]][0]

    def __len__(self):
        return len(self.indices)


class AnomalousCIFAR10ImageDataset(Dataset):
    def __init__(self, nominal_class, transform, train=True):
        self.data = torchvision.datasets.CIFAR10(
            'datasets',
            train=train,
            download=True,
            transform=transform
        )

        self.indices = torch.where(torch.as_tensor(self.data.targets) != nominal_class)[0]

    def __getitem__(self, item):
        return self.data[self.indices[item % len(self.indices)]][0]

    def __len__(self):
        return len(self.indices)
