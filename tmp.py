import torch

X = torch.Tensor([[1, 2, 3],
                  [2, 3, 4],
                  [5, 4, 1]]).cuda()
print(X)
print(X.argmax(dim=1))