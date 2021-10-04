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
group_size       = 64
population       = []
#--------------------------------------

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 랜덤 시드 고정
torch.manual_seed(777)

# GPU 사용 가능일 경우 랜덤 시드 고정
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

INPUT  = [0 for x in range(int((width/chwdth) * (height/chhght)))]    #activation layer
shape = (1,1,20,20)
for i in range(0, group_size):
    population.append(net.CNN.to(device))

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
            
            x_data = x_data.to(device)
            prediction = model(x_data)
            direction = torch.argmax(prediction)
            timepassed += 1
            if timepassed % 10 == 0:
                taillen += 1
#mainloop ends
        prediction.fit  = timepassed
        trail               = []
        direction           = random.randrange(0,3)
        timepassed          = 0
        taillen             = 1
        posx                = (width/2) - chwdth
        posy                = height/2 - chhght
        indiv_counter      += 1

    parents = sorted(population, key=lambda individual : individual.fit, reverse= True)
    # parents = sorted(population, key=lambda individual : individual.fit)
    parents = parents[0:int(group_size/8)]
    ###debug
    # for c in range(len(parents)):
    #     print(parents[c].fit)
    ###debug
    elite = parents[0]
    bestfit = elite.fit

    population = obj.MPX(parents, group_size)
    population[0] = elite
    print("bestfit...", bestfit)
    generation += 1