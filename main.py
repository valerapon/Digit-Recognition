import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from PIL import Image

# X = torch.tensor([1, 2])
# Y = torch.randn(5, 5, device='cuda', dtype=float)

# print(X)
# print(Y)
# print(os.getcwd())
# torchvision.datasets.MNIST(root=os.getcwd(), train=True, download=True)

train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
                root=os.getcwd(),
                train=True,
                download=True
        ),
        batch_size=64,
        shuffle=True
)

print(train_loader)