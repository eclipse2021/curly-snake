import torch
import torch.nn as nn
import random
from torch.nn.modules import padding

# model architecture #
# shape = (1, 1, 20, 20)
# x = torch.zeros(shape)
# conv1 = nn.Conv2d(1, 8, 3)
# conv2 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
# pool = nn.MaxPool2d(2)

# out = conv1(x)
# out = pool(out)
# out = conv2(out)
# out = pool(out)
# out = out.view(out.size(0), -1)
# print(out.shape)


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.fit = 0
        # 첫번째층

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, kernel_size=3, stride=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        # 두번째층

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        # 전결합층 4x4x8 inputs -> 4 outputs
        self.fc = torch.nn.Linear(4 * 4 * 8, 4, bias=True)

        # 전결합층 한정으로 가중치 초기화
        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)   # 전결합층을 위해서 Flatten
        out = self.fc(out)
        return out

if __name__ == "__ma_":
    while True:
        model = CNN()
        model_offs = CNN()
        select_start = (random.randint(0, 2),random.randint(0,2))
        select_end   = (random.randint(select_start[0],3),random.randint(select_start[1],3))
        print(model.layer1[0].weight[0][0])
        gene = model.layer1[0].weight[0][0][:, select_start[0]:select_end[0]][select_start[1]:select_end[1]]
        print(gene)
        print(model_offs.layer1[0].weight[0])
        with torch.no_grad():
            model_offs.layer1[0].weight[0][0][:, select_start[0]:select_end[0]][select_start[1]:select_end[1]] = gene
        print(model_offs.layer1[0].weight[0])
        input("")
        print("\n\n\n\n")
if __name__ == "__main__":
    model = CNN()
    print(model.layer2[0].weight[0][0])
