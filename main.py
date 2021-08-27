import sys, pygame
import random
import obj

#constants
sqrt_2 = 2 ** 0.5

pygame.init()

bg = pygame.image.load("./img/bg.png")
ch = pygame.image.load("./img/p.png")

size = width, height = 720, 720

chsize = ch.get_rect().size
chwdth = chsize[0]
chhght = chsize[1]
posx = (width/2) - chwdth
posy = height/2 - chhght

trail = []
direction = random.randrange(0,3)

timeinst = pygame.time.Clock()
timepassed = 0
speed = 60

taillen = 1

screen = pygame.display.set_mode(size) # screen

font5 = pygame.font.SysFont('malgungothicsemilight',30) # font


individual_alive = True
group_size = 1000
population = []
for _ in range(0, group_size):
    population.append(obj.Sequential([
        obj.ReLU(400,16),
        obj.ReLU(16,16),
        obj.ReLU(16,4),
        obj.softmax(4)
    ]))

INPUT  = [0 for x in range(int((width/chwdth) * (height/chhght)))]    #activation layer

generation = 0
while True:
    print("gen:" + str(generation))
    indiv_counter = 0
    for individual in population:
        
        if indiv_counter%100 == 0:
            print("ind:" + str(indiv_counter))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        while True:
            # delta_time = timeinst.tick(speed)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
            
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
            
            screen.blit(bg, (0,0))


            loops_done = 0
            while loops_done < taillen:
                INPUT[int(trail[loops_done][0]/chwdth + (trail[loops_done][1]/chhght) * (height/chhght))] = 1
                screen.blit(ch, (trail[loops_done][0], trail[loops_done][1]))
                loops_done +=  1

            current_score = font5.render(str(taillen), True ,(0,0,0))
            # current_indiv = font5.render(str(indv_count), True, (220,220,220))
            screen.blit(ch, (posx, posy))
            screen.blit(current_score,(posx, posy))
            # screen.blit(current_indiv,(0,0))


            OUT = individual.forward(INPUT)
            direction = OUT.index(max(OUT))
            timepassed += 1
            if timepassed % 10 == 0:
                taillen += 1
            pygame.display.update()
        
        individual.fitness = timepassed
        trail = []
        direction = random.randrange(0,3)
        timepassed = 0
        taillen = 1
        posx = (width/2) - chwdth
        posy = height/2 - chhght
        indiv_counter += 1

    parents = sorted(population, key=lambda individual : individual.fitness, reverse= True)
    parents = parents[:int(len(parents)/2)]
    print('best fit:' + str(parents[0].fitness))
    obj.TNC(parents)
    generation += 1


    #ranking = []    #전체 적합도 결과

    #for i in range(0, len(population)):
    #    ranking.append(population[i].fitness)
#
    #print(str(ranking))
    #ranking_sorted = ranking
    #ranking_sorted.sort()    #적합도 정렬
    #print(ranking_sorted)
#
    #top_fits_key = []    #중복 제거된 상위 적합도
#
    #pointer = len(ranking_sorted) - 1
    #while pointer > len(ranking_sorted)/2 - 1:
    #    if ranking_sorted[pointer] not in top_fits_key:
    #        top_fits_key.append(ranking_sorted[pointer])
    #    pointer -= 1
#
    #print(top_fits_key)
#
    #parents = []
    #for pointer in range(0, len(population)):    #상위 개체들
    #    if population[pointer].fitness in top_fits_key:
    #        parents.append(population[pointer])


    # nextgen1 = []
    # nextgen2 = []
    
    # for unit in parents:
    #     nextgen1.append(unit)
    # for unit in parents:
    #     nextgen2.append(unit)

    # nextgen1 = obj.crossover(nextgen1)
    # nextgen2 = obj.crossover(nextgen2)
    
    # population = nextgen1 + nextgen2
    # generation += 1