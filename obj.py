import numpy as np
#import decimal
import random

from numpy.lib.function_base import percentile

constants_e = 2.71828
c_e = str(np.exp(1))

class Sequential:
    def __init__(self, sequence: list) -> None:
        self.sequence = sequence
        self.fit = 0
    
    def forward(self, IN):
        for layer in self.sequence:
            IN = layer.activate(IN)
        return IN

class ReLU:
    def __init__(self, in_features, out_features) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.w = [[0 for x in range(in_features)] for y in range(out_features)]
        self.b = [0 for x in range(out_features)]
        for index_y in range(out_features):
            for index_x in range(self.in_features):
                self.w[index_y][index_x] = random.uniform(-1, 1)
        for index_x in range(out_features):
            self.b[index_x] = random.uniform(-1, 1)

    
    def relu(slef, x):
        return max(0,x)

    def activate(self, IN: list) -> list:
        OUT = [0 for x in  range(self.out_features)]
        neurons_activated = 0
        while neurons_activated < self.out_features:
            temp_neuron = 0
            # temp_neuron += (IN[index] * self.w[neurons_activated][index] for index in range(self.in_features))
            # temp_neuron += self.b[neurons_activated]

            for index in range(self.in_features):
                temp_neuron += IN[index] * self.w[neurons_activated][index]
            temp_neuron += self.b[neurons_activated]
            OUT[neurons_activated] = self.relu(temp_neuron)
            neurons_activated += 1
        return OUT

class softmax:
    def __init__(self, dimensions) -> None:
        self.dimensions = dimensions
    def activate(self, IN: list) -> list:
        c = max(IN)
        IN = [x - c for x in IN]
        sum = 0
        for x in IN:
            sum += np.exp(x)
        OUT = [np.exp(x)/sum for x in IN]
        return OUT




def MPX(population: list, population_size: int, mutant_possibility = 0.05):
    offsprings = []
    while len(offsprings) < population_size:
        random.shuffle(population)
        left  = population[0]
        right = population[0]
        if len(left.sequence) == len(right.sequence):
            for j in range(len(left.sequence)):
                left_layer  =  left.sequence[j]
                right_layer = right.sequence[j]

                try:
                    getw = left_layer.w
                except AttributeError as e:
                    #print(e)
                    break
                
                #weight cx starts here + mutant
                for k in range(len(left_layer.w)):
                    for l in range(len(left_layer.w[k])):
                        if random.random() < 0.5:
                            left_weight  =  left_layer.w[k][l]
                            right_weight = right_layer.w[k][l]
                            left_layer.w[k][l]  = right_weight
                            right_layer.w[k][l] =  left_weight

                    if random.random() <= mutant_possibility:
                        left_layer.w[k] = [x * random.uniform(-1,1) for x in left_layer.w[k]]
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
                if random.random() <= mutant_possibility:
                    left_layer.b = [x * random.uniform(-1,1) for x in left_layer.b]
        else:
            print("cannot crossover between two individuals that has different sequences")
        offsprings.append(left)
        offsprings.append(right)
        print("\n" * 150, "generating...", len(offsprings)/population_size * 100 , "%")
    return offsprings



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
                
                #weight cx starts here + mutant
                for k in range(len(left_layer.w)):
                    for l in range(len(left_layer.w[k])):
                        if random.random() < 0.5:
                            left_weight  =  left_layer.w[k][l]
                            right_weight = right_layer.w[k][l]
                            left_layer.w[k][l]  = right_weight
                            right_layer.w[k][l] =  left_weight

                    if random.random() <= mutant_possibility:
                        left_layer.w[k] = [x * random.uniform(-1,1) for x in left_layer.w[k]]
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
                if random.random() <= mutant_possibility:
                    left_layer.b = [x * random.uniform(-1,1) for x in left_layer.b]
        else:
            print("cannot crossover between two individuals that has different sequences")
        population.append(left)
        population.append(right)

    return population
