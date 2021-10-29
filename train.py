import random

from torch.nn import modules
import obj
import torch
import net

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
group_size       = 32
population       = []
mutation_possibility_0 = 0.05
mutation_possibility = mutation_possibility_0
desired_fit = 400
#--------------------------------------

INPUT  = [0 for x in range(int((width/chwdth) * (height/chhght)))]    #activation layer
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
                INPUT[int(trail[0][0]/chwdth + (trail[0][1]/chhght)*(height/chhght))] = 0
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
                INPUT[int(trail[tails_found][0]/chwdth + (trail[tails_found][1]/chhght) * (height/chhght))] = 1
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
    parents = parents[0:int(group_size/4)]
    ###debug
    # for c in range(len(parents)):
    #     print(parents[c].fit)
    ###debug
    elite = parents[0]
    bestfit = elite.fit
    
    mutations_occured = 0
    offsprings = []
    while len(offsprings) < group_size - len(parents):
        random.shuffle(parents)
        model1 = parents[0]
        model2 = parents[1]
        off1 = net.CNN()
        off2 = net.CNN()
        if random.uniform(0,1) <= mutation_possibility:
            offsprings.append(off1)
            offsprings.append(off2)
            mutations_occured += 1
            continue
        for q in range(len(model1.layer1[0].weight)):
            for w in range(len(model1.layer1[0].weight[q])):
                for e in range(len(model1.layer1[0].weight[q][w])):
                    for r in range(len(model1.layer1[0].weight[q][w][e])):
                        left  = model1.layer1[0].weight[q][w][e][r].item()
                        right = model2.layer1[0].weight[q][w][e][r].item() 
                        if random.uniform(0,1) < 0.4:
                           with torch.no_grad():
                                off1.layer1[0].weight[q][w][e][r] = right
                                off2.layer1[0].weight[q][w][e][r] =  left
                        else:
                            with torch.no_grad():
                                off1.layer1[0].weight[q][w][e][r] =  left
                                off2.layer1[0].weight[q][w][e][r] = right
        for q in range(len(model1.layer1[0].bias)):
            left = model1.layer1[0].bias[q].item()
            right = model2.layer1[0].bias[q].item()
            if random.uniform(0,1) < 0.4:
                with torch.no_grad():
                    off1.layer1[0].bias[q] = right
                    off2.layer1[0].bias[q] = left
            else:
                with torch.no_grad():
                    off1.layer1[0].bias[q] = left
                    off2.layer1[0].bias[q] = right
        for q in range(len(model1.layer2[0].weight)):
            for w in range(len(model1.layer2[0].weight[q])):
                for e in range(len(model1.layer2[0].weight[q][w])):
                    for r in range(len(model1.layer2[0].weight[q][w][e])):
                        left  = model1.layer2[0].weight[q][w][e][r].item()
                        right = model2.layer2[0].weight[q][w][e][r].item() 
                        if random.uniform(0,1) < 0.4:
                            with torch.no_grad():
                                off1.layer2[0].weight[q][w][e][r] = right
                                off2.layer2[0].weight[q][w][e][r] = left
                        else:
                            with torch.no_grad():
                                off1.layer2[0].weight[q][w][e][r] = left
                                off2.layer2[0].weight[q][w][e][r] = right
        for q in range(len(model1.layer2[0].bias)):
            left = model1.layer1[0].bias[q].item()
            right = model2.layer1[0].bias[q].item()
            if random.uniform(0,1) < 0.4:
                with torch.no_grad():
                    off1.layer1[0].bias[q] = right
                    off2.layer1[0].bias[q] = left
            else:
                with torch.no_grad():
                    off1.layer1[0].bias[q] = left
                    off2.layer1[0].bias[q] = right

        for q in range(len(model1.fc.weight)):
            for w in range(len(model1.fc.weight[q])):
                left  = model1.fc.weight[q][w]
                right = model2.fc.weight[q][w]
                if random.uniform(0,1) < 0.4:
                    with torch.no_grad():
                        off1.fc.weight[q][w] = right
                        off2.fc.weight[q][w] = left
                else:
                    with torch.no_grad():
                        off1.fc.weight[q][w] = left
                        off2.fc.weight[q][w] = right
        for q in range(len(model1.fc.bias)):
            left = model1.fc.bias[q].item()
            right = model2.fc.bias[q].item()
            if random.uniform(0,1) < 0.4:
                with torch.no_grad():
                    off1.fc.bias[q] = right
                    off2.fc.bias[q] = left
            else:
                with torch.no_grad():
                    off1.fc.bias[q] = left
                    off2.fc.bias[q] = left
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
