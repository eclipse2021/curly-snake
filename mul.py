import torch

shape = (3,3)
x = torch.FloatTensor(
    [[2.,2.],
     [2.,2.]])
print(x)
y = torch.rand(x.shape)
print(y)
print(x * y)
