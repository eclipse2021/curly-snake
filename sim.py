import net
import random
from copy import deepcopy

def showlist(list_arg, space = 3):
    for x in list_arg:
        if x <= 9:
            print(x, end = ' '*2)
        if x >= 10:
            print(x, end = ' ')
    print()

li1 = [[x for x in range(10)] for y in range(2)]
li2 = [[x for x in range(10,20)] for y in range(2)]

point = random.randint(0, len(li1[0]))

off1 = [[]]
off2 = [[]]
off1[0] = deepcopy(li1[0])
off2[0] = deepcopy(li2[0])

off1[0][:point] = li2[0][:point]
off2[0][:point] = li1[0][:point]
showlist(li1[0])
showlist(li2[0])
print('|\nv')
showlist(off1[0])
showlist(off2[0])

model1 = net.CNN()
print(model1.layer1[0].weight[0][0])
