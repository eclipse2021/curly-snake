import obj
import numpy as np

model = obj.Sequential([
    obj.ReLU(10,3),
    obj.ReLU(3,4),
    obj.softmax(4)
])

population = []

for i in range(2):
    population.append(obj.Sequential([obj.ReLU(10,3),obj.ReLU(3,4),obj.softmax(4)]))
te = [[10], [0]]

def is_list(li:list):
    deepness_counter = 0
    thismightbe_list = li
    while str(type(thismightbe_list[0])) == "<class 'list'>":
        deepness_counter += 1
        print("Deepness = ", deepness_counter) 
        thismightbe_list = thismightbe_list[0]
    return deepness_counter
te = [1]


