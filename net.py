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

if __name__ == "__main__0":
    weights_counter = 0#debug
    weights_crossed = 0#debug

    shape = (1,1,20,20)
    x = torch.randn(shape)
    model1 = CNN()
    model2 = CNN()
    print(model1.layer1[0].weight[0][0][0][0].item())
    with torch.no_grad():
        model1.layer1[0].weight[0][0][0][0] = 0
    print("weight assigned in progress")
    #print(model1.layer1[0].weight)#layer1[0] = conv layer , layer1[1] = relu , layer1[2] = maxpool
    for q in range(len(model1.layer1[0].weight)):
        for w in range(len(model1.layer1[0].weight[q])):
                for e in range(len(model1.layer1[0].weight[q][w])):
                    for r in range(len(model1.layer1[0].weight[q][w][e])):
                        weights_counter += 1
                        print("w1:", model1.layer1[0].weight[q][w][e][r])
                        print("w2:", model2.layer1[0].weight[q][w][e][r])
                        if random.uniform(0,1) < 0.4:
                            left  = model1.layer1[0].weight[q][w][e][r].item()
                            right = model2.layer1[0].weight[q][w][e][r].item()
                            with torch.no_grad():
                                model1.layer1[0].weight[q][w][e][r] = right
                                model2.layer1[0].weight[q][w][e][r] =  left
                            print("weight crossed:")
                            print("w1:", model1.layer1[0].weight[q][w][e][r])
                            print("w2:", model2.layer1[0].weight[q][w][e][r])
                            weights_crossed += 1

                        print("\n\n")
    
    print("total weights:", weights_counter)
    print("weigths crossed:", weights_crossed)
if __name__ == "__main__0":
    weights_counter = 0#debug
    weights_crossed = 0#debug

    shape = (1,1,20,20)
    x = torch.randn(shape)
    model1 = CNN()
    model2 = CNN()
    print(model1.layer2[0].weight[0])
    for q in range(len(model1.layer2[0].weight)):
        for w in range(len(model1.layer2[0].weight[q])):
            for e in range(len(model1.layer2[0].weight[q][w])):
                for r in range(len(model1.layer2[0].weight[q][w][e])):
                    print("w1:", model1.layer2[0].weight[q][w][e][r])
                    print("w2:", model1.layer2[0].weight[q][w][e][r])
                    if random.uniform(0,1) < 0.4:
                        left  = model1.layer2[0].weight[q][w][e][r].item()
                        right = model2.layer2[0].weight[q][w][e][r].item()
                        with torch.no_grad():
                            model1.layer2[0].weight[q][w][e][r] = right
                            model2.layer2[0].weight[q][w][e][r] = left
                            print("weight crossed:")
                            print("w1:", model1.layer2[0].weight[q][w][e][r])
                            print("W2:", model2.layer2[0].weight[q][w][e][r])
                            weights_crossed += 1
                    weights_counter += 1
                    print("\n\n")
    print("total weights = ", weights_counter)
    print("weights crossed =", weights_crossed)
if __name__ == "__main__0":
    weights_counter = 0
    weights_crossed = 0
    shape = (1,1,20,20)
    model1 = CNN()
    model2 = CNN()
    print(model1.fc.weight[0][0])

    for q in range(len(model1.fc.weight)):
        for w in range(len(model1.fc.weight[q])):
            print("w1", model1.fc.weight[q][w])
            print("w2", model2.fc.weight[q][w])
            if random.uniform(0,1) < 0.4:
                left = model1.fc.weight[q][w]
                right = model2.fc.weight[q][w]
                with torch.no_grad():
                    model1.fc.weight[q][w] = right
                    model2.fc.weight[q][w] = left
                    print("weight crosssed:")
                    print("w1:", model1.fc.weight[q][w])
                    print("w2:", model2.fc.weight[q][w])
                    weights_crossed += 1
            weights_counter += 1
            print("\n\n")
    print("total weights:" , weights_counter)
    print("weights crossed:", weights_crossed)
if __name__ == "__main__":
    weights_counter = 0
    weights_crossed = 0
    shape = (1,1,20,20)
    model1 = CNN()
    model2 = CNN()
    for q in range(len(model1.layer1[0].bias)):
        print(model1.layer2[0].bias[q])
