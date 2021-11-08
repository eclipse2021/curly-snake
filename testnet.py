import torch


agent = torch.nn.Sequential(
        torch.nn.Conv2d(1, 5, kernel_size=3, stride=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=2, stride=2),
        torch.nn.Conv2d(5, 5, kernel_size=3, stride=1, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=2, stride=2),
        torch.nn.Flatten())

x = torch.zeros(1,1,20,20)
y = agent(x)
print(y.size())
