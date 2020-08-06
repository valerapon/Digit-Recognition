import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
from matplotlib import pyplot as plt

class Network(nn.Module):
        def __init__(self):
                super(Network, self).__init__()
                self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
                self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
                self.conv2_drop = nn.Dropout2d()
                self.fc1 = nn.Linear(320, 50)
                self.fc2 = nn.Linear(50, 10)

        def forward(self, x):
                x = F.relu(F.max_pool2d(self.conv1(x), 2))
                x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
                x = x.view(-1, 320)
                x = F.relu(self.fc1(x))
                x = F.dropout(x, training=self.training)
                x = self.fc2(x)
                return F.log_softmax(x)
                

if __name__ == '__main__':
        train_loader = torch.utils.data.DataLoader(
                torchvision.datasets.MNIST(
                        root=os.getcwd(),
                        train=True,
                        download=True,
                        transform=torchvision.transforms.Compose([
                                        torchvision.transforms.ToTensor()
                                ])
                ),
                batch_size=64,
                shuffle=True
        )

        test_loader = torch.utils.data.DataLoader(
                torchvision.datasets.MNIST(
                        root=os.getcwd(),
                        train=False,
                        download=True,
                        transform=torchvision.transforms.Compose([
                                        torchvision.transforms.ToTensor()
                                ])
                ),
                batch_size=1000,
                shuffle=True
        )

        Net = Network()
        optimizer = optim.SGD(Net.parameters(), lr=0.01, momentum=0.5)


