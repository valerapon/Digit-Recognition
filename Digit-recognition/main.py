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

batch_size = 50
epoch_num = 20

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
                # x = F.relu(F.max_pool2d(self.conv2(x), 2))
                x = x.view(-1, 320)
                x = F.relu(self.fc1(x))
                x = F.dropout(x, training=self.training)
                x = self.fc2(x)
                return F.log_softmax(x, -1)
                

def train(network, train_loader, optimizer, epoch):
        network.train()
        for batch_idx, (data, target) in enumerate(train_loader):

                data = data.cuda()
                target = target.cuda()

                def closure():
                        optimizer.zero_grad()
                        output = network.forward(data)
                        loss = F.cross_entropy(output, target)
                        loss.backward()   
                        if batch_idx % 100 == 0:
                                print('Train epoch: %s, {%s}, Loss: %s' % (epoch, batch_idx, loss.item()))
                        return loss
                optimizer.step(closure)



def test(network, test_loader):
        test_loss = 0
        correct = 0

        network.eval()
        with torch.no_grad():
                for batch_idx, (data, target) in enumerate(test_loader):

                        data = data.cuda()
                        target = target.cuda()

                        output = network.forward(data)
                        # test_loss += F.cross_entropy(output, train).item()
                        # print(output)
                        pred = output.argmax(dim=1)
                        correct += (pred == target).sum().item()
                print('TOTAL:', correct / len(test_loader.dataset))

                # test_loss /= len(test_loader.dataset)
                # print(test_loader)
                        # pred = output.data.max(1, keepdim=True)[1]
                        # correct += pred.eq(target.data.view_as(pred)).sum()



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
                batch_size=batch_size,
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

        Net = Network().to(device='cuda')
        optimizer = optim.Adagrad(Net.parameters(), lr=0.01)

        for i in range(epoch_num):
                train(Net, train_loader, optimizer, i)

        test(Net, test_loader)

    






