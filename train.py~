import random

from torch.nn import modules
import obj
import torch
import net

from copy import deepcopy
#constants------------------------------
sqrt_2           = 2 ** 0.5

width            = 720    #가로20칸
height           = 720    #세로20칸

chwdth           = 36
chhght           = 36
posx             = (width/2) - chwdth
posy             = height/2 - chhght

trail            = []
direction        = random.randrange(0,3)
timepassed       = 0
taillen          = 1

individual_alive = True
group_size       = 64
population       = []
mutation_possibility_0 = 0.05
mutation_possibility = mutation_possibility_0
desired_fit = 400
global_bestfit = 0

MODEL_PATH = "./models/"
#--------------------------------------

shape = (1,1,20,20)
for i in range(0, group_size):
    population.append(net.CNN())

generation = 0
while True:
    print("gen:" + str(generation))
    indiv_counter = 0
    for model in population:
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
            prediction = model(x_data)
            direction = torch.argmax(prediction)
            timepassed += 1
            if timepassed % 10 == 0:
                taillen += 1
#mainloop ends
        model.fit  = timepassed
        trail               = []
        direction           = random.randrange(0,3)
        timepassed          = 0
        taillen             = 1
        posx                = (width/2) - chwdth
        posy                = height/2 - chhght
        indiv_counter      += 1

    parents = sorted(population, key=lambda model : model.fit, reverse= True)
    # parents = sorted(population, key=lambda individual : individual.fit)
    parents = parents[0:int(group_size/8)]
    ###debug
    # for c in range(len(parents)):
    #     print(parents[c].fit)
    ###debug
    fittest = parents[0]
    bestfit = fittest.fit

    if bestfit < global_bestfit:
        global_bestfit = bestfit
        torch.save(fittest, MODEL_PATH + "model.pt")

    if bestfit == 400:
        #terminal condition
        break

    if generation >= 1000:
        #terminal condition
        break
        
    mutations_occured = 0
    if random.uniform(0,0.5) <= mutation_possibility:
        parents[random.randint(1,len(parents) - 1)] = net.CNN()
        mutations_occured += 1
    
    offsprings = []
    random.shuffle(parents)
    while len(offsprings) < group_size - len(parents):
        model1 = parents[len(offsprings)%len(parents)]
        model2 = parents[len(offsprings)%len(parents)+1]
        off1 = net.CNN()
        off2 = net.CNN()
       # if random.uniform(0,1) <= mutation_possibility:
       #     offsprings.append(off1)
       #     offsprings.append(off2)
       #     mutations_occured += 1
       #     continue
        for q in range(len(model1.layer1[0].weight)):
            for w in range(len(model1.layer1[0].weight[q])):
                for e in range(len(model1.layer1[0].weight[q][w])):
                    point_start = (random.randint(0,2), random.randint(0,2))
                    point_end = (random.randint(point_start[0],3), random.randint(point_start[1], 3))
                    with torch.no_grad():
                        off1.layer1[0].weight[q][w] = deepcopy(model1.layer1[0].weight[q][w])
                        off1.layer1[0].weight[q][w][:, point_start[0]:point_end[0]][point_start[1]:point_end[1]] = model2.layer1[0].weight[q][w][:, point_start[0]:point_end[0]][point_start[1]:point_end[1]]
                        off2.layer1[0].weight[q][w] = deepcopy(model2.layer1[0].weight[q][w])
                        off2.layer1[0].weight[q][w][:, point_start[0]:point_end[0]][point_start[1]:point_end[1]] = model1.layer1[0].weight[q][w][:, point_start[0]:point_end[0]][point_start[1]:point_end[1]]
        for q in range(len(model1.layer1[0].bias)):
            point = random.randint(0,len(model1.layer1[0].bias))
            with torch.no_grad():
                off1.layer1[0].bias = deepcopy(model1.layer1[0].bias)
                off2.layer1[0].bias = deepcopy(model2.layer1[0].bias)
                off1.layer1[0].bias[:point] = model2.layer1[0].bias[:point]
                off2.layer1[0].bias[:point] = model1.layer1[0].bias[:point]
        for q in range(len(model1.layer2[0].weight)):
            for w in range(len(model1.layer2[0].weight[q])):
                for e in range(len(model1.layer2[0].weight[q][w])):
                    point_start = (random.randint(0,2), random.randint(0,2))
                    point_end = (random.randint(point_start[0],3), random.randint(point_start[1], 3))
                    with torch.no_grad():
                        off1.layer2[0].weight[q][w] = deepcopy(model1.layer2[0].weight[q][w])
                        off1.layer2[0].weight[q][w][:, point_start[0]:point_end[0]][point_start[1]:point_end[1]] = model2.layer2[0].weight[q][w][:, point_start[0]:point_end[0]][point_start[1]:point_end[1]]
                        off2.layer2[0].weight[q][w] = deepcopy(model2.layer2[0].weight[q][w])
                        off2.layer2[0].weight[q][w][:, point_start[0]:point_end[0]][point_start[1]:point_end[1]] = model1.layer2[0].weight[q][w][:, point_start[0]:point_end[0]][point_start[1]:point_end[1]]
        for q in range(len(model1.layer2[0].bias)):
            point = random.randint(0,len(model1.layer2[0].bias))
            with torch.no_grad():
                off1.layer2[0].bias = deepcopy(model1.layer2[0].bias)
                off2.layer2[0].bias = deepcopy(model2.layer2[0].bias)
                off1.layer2[0].bias[:point] = model2.layer2[0].bias[:point]
                off2.layer2[0].bias[:point] = model1.layer2[0].bias[:point]
        for q in range(len(model1.fc.weight)):
            point = random.randint(0,len(model1.fc.weight[q]))
            with torch.no_grad():
                off1.fc.weight[q] = deepcopy(model1.fc.weight[q])
                off2.fc.weight[q] = deepcopy(model2.fc.weight[q])
                off1.fc.weight[:point] = model2.fc.weight[:point]
                off2.fc.weight[:point] = model1.fc.weight[:point]
        for q in range(len(model1.fc.bias)):
            point = random.randint(0,len(model1.fc.bias))
            with torch.no_grad():
                off1.fc.bias = deepcopy(model1.fc.bias)
                off2.fc.bias = deepcopy(model2.fc.bias)
                off1.fc.bias[:point] = model2.fc.bias[:point]
                off2.fc.bias[:point] = model1.fc.bias[:point]
        offsprings.append(off1)
        offsprings.append(off2)
    population = offsprings + parents
    if len(population) == group_size:
        print("generation completed: total", len(population),"/", len(offsprings), "+", len(parents))
    else:
        print("err: individual loss: total", len(population),"/", len(offsprings), "+", len(parents))
        input()
    print("bestfit...", bestfit)
    print("mutation...:", mutation_possibility * 100, "%")
    print("mutation occured:", mutations_occured, "\n\n")
    mutation_possibility = mutation_possibility_0 * ((bestfit/desired_fit - 1) ** 4)
    generation += 1
