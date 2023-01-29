import torch
import torchvision
import torch.nn as nn


class ConvNet_CIFAR10(nn.Module):
    def __init__(self):
        super(ConvNet_CIFAR10, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten()
        )

        self.fc_layers = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(512*4*4, 2048),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 10)
        )