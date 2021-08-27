import random
import obj

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
group_size       = 500
population       = []
#--------------------------------------

# class SoftmaxModel(obj.Individual):
#     def __init__(self) -> None:
#         super().__init__()
#     pass

INPUT  = [0 for x in range(int((width/chwdth) * (height/chhght)))]    #activation layer

for i in range(0, group_size):
    population.append(obj.Sequential([
        obj.ReLU(400,16),
        obj.ReLU(16,16),
        obj.ReLU(16,4),
        obj.softmax(4)
    ]))

generation = 0
while True:
    print("gen:" + str(generation))
    indiv_counter = 0
    for individual in population:
        if indiv_counter%100 == 0:
            print("ind:" + str(indiv_counter))
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
                tails_found += 1

            OUT = individual.forward(INPUT)
            direction = OUT.index(max(OUT))
            timepassed += 1
            if timepassed % 10 == 0:
                taillen += 1
#mainloop ends
        individual.fitness  = timepassed
        trail               = []
        direction           = random.randrange(0,3)
        timepassed          = 0
        taillen             = 1
        posx                = (width/2) - chwdth
        posy                = height/2 - chhght
        indiv_counter      += 1

    parents = sorted(population, key=lambda individual : individual.fitness, reverse= True)
    parents = parents[:1000]
    print('best fit:' + str(parents[0].fitness))

    obj.TNC(parents)
    generation += 1

    # nextgen1 = []
    # nextgen2 = []
    
    # for unit in parents:
    #     nextgen1.append(unit)
    # for unit in parents:
    #     nextgen2.append(unit)

    # nextgen1    = obj.crossover_active(nextgen1)
    # nextgen2    = obj.crossover_active(nextgen2)
    
    # population  = nextgen1 + nextgen2
    # generation += 1