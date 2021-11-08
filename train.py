import random

from torch.nn import modules
import obj
import torch
import net

import pygad.torchga
import pygad

from copy import deepcopy
#constants------------------------------
sqrt_2             = 2 ** 0.5
                   
width              = 720    #가로20칸
height             = 720    #세로20칸
                   
chwdth             = 36
chhght             = 36
posx               = (width/2) - chwdth
posy               = height/2 - chhght
                   
trail              = []
direction          = random.randrange(0,3)
timepassed         = 0
taillen            = 1
                   
individual_alive   = True
group_size         = 64
num_generations    = 50000
num_parents_mating = 5

MODEL_PATH = "./models/"
#--------------------------------------

def fitness_func(solution, sol_idx):
    global torch_ga, agent, shape, width, height, chhght, chwdth, posy, posx, trail, direction, timepassed, taillen
    timepassed = 0
    x_data = torch.zeros(shape)
    while True:    #main loop
        if direction == 0:
            posy -= chhght
        if direction == 1:
            posy += chhght
        if direction == 2:
            posx -= chhght
        if direction == 3:
            posx += chhght
        
        trail.append([posx, posy])
        
        if len(trail) - 1 > taillen:
            x_data[0][0][int(trail[0][1]/chhght)][int(trail[0][0]/chwdth)] = 0
            del trail[0]
        
        if posx < 0:
            break
        if posx > width - chwdth:
            break
        if posy < 0:
            break
        if posy > height - chhght:
            break
        
        if len(trail) > 1:
            if [posx, posy] in trail[:len(trail) - 1]:
                break
        
        tails_found = 0
        while tails_found < taillen:
            x_data[0][0][int(trail[tails_found][1]/chhght)][int(trail[tails_found][0]/chwdth)] = 1
            tails_found += 1
        x_data[0][0][int(posx/chhght)][int(posy/chwdth)] = 4
        prediction = pygad.torchga.predict(model = agent, solution = solution, data = x_data)
        direction = torch.argmax(prediction)
        timepassed += 1
        if timepassed % 10 == 0:
            taillen += 1
    trail               = []
    direction           = random.randrange(0,3)
    taillen             = 1
    posx                = (width/2) - chwdth
    posy                = height/2 - chhght
    return timepassed

def callback_generation(ga_instance):
    print("Generation" + str(ga_instance.generations_completed), end = '|')
    print("Fitness=" + str(ga_instance.best_solution()[1]))

agent = torch.nn.Sequential(
        torch.nn.Conv2d(1, 8, kernel_size=3, stride=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=2, stride=2),
        torch.nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=2, stride=2),
        torch.nn.Flatten(),
        torch.nn.Linear(4 * 4 * 8, 4, bias=True),
        torch.nn.ReLU(),
        torch.nn.Linear(4,4, bias = True))


torch_ga = pygad.torchga.TorchGA(model=agent, num_solutions = 10)

shape = (1,1,20,20) # input image size

initial_population = torch_ga.population_weights

ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       initial_population=initial_population,
                       fitness_func=fitness_func,
                       on_generation=callback_generation)

ga_instance.run()

