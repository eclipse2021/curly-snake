import obj
import numpy as np
import random

model = obj.Sequential([
    obj.ReLU(10,3),
    obj.ReLU(3,4),
    obj.softmax(4)
])

population = []
population_size = 1024

for i in range(population_size):
    population.append(obj.Sequential([obj.ReLU(10,3),obj.ReLU(3,4),obj.softmax(4)]))

def get_depth(li:list):
    deepness_counter = 0
    thismightbe_list = li
    while str(type(thismightbe_list[0])) == "<class 'list'>":
        deepness_counter += 1
        print("Deepness = ", deepness_counter) 
        thismightbe_list = thismightbe_list[0]
    return deepness_counter

print(population[0].sequence[0].w[0])
print(population[1].sequence[0].w[0])


random.shuffle(population)
for i in range(int(len(population)/2)):
    left  = population.pop(0)
    right = population.pop(0)
    if len(left.sequence) == len(right.sequence):
        for j in range(len(left.sequence)):
            left_layer  =  left.sequence[j]
            right_layer = right.sequence[j]
            
            try:
                getw = left_layer.w
            except AttributeError as e:
                #print(e)
                break
            
            #weight cx starts here
            for k in range(len(left_layer.w)):
                for l in range(len(left_layer.w[k])):
                    if random.random() < 0.5:
                        left_weight  =  left_layer.w[k][l]
                        right_weight = right_layer.w[k][l]
                        left_layer.w[k][l]  = right_weight
                        right_layer.w[k][l] =  left_weight
            #bias cx starts here
            
            try:
                getw = left_layer.b
            except AttributeError as e:
                #print(e)
                break
            
            for k in range(len(left_layer.b)):
                if random.random() < 0.5:
                    left_bias = left_layer.b[k]
                    right_bias = right_layer.b[k]
                    left_layer.b[k] = right_bias
                    right_layer.b[k] = left_bias  
    else:
        print("cannot crossover between two individuals that has different sequences")
    population.append(left)
    population.append(right)

print("\n\n",population[0].sequence[0].w[0])
print(population[1].sequence[0].w[0])
