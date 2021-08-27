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

INPUT  = [0 for x in range(int((width/chwdth) * (height/chhght)))]    #activation layer

for _ in range(0, group_size):
    population.append(obj.Individual())

generation = 0
while True:
    print("gen:" + str(generation))
    indiv_counter = 0
    for individual in population:
        if indiv_counter%100 == 0:
            print("ind:" + str(indiv_counter))
        while True:    #main loop
            up     = 0
            down   = 0
            left   = 0
            right  = 0
            if direction == 0:
                posy -= chhght
                up    = 1
            if direction == 1:
                posy += chhght
                down  = 1
            if direction == 2:
                posx -= chhght
                left  = 1
            if direction == 3:
                posx += chhght
                right = 1

            INPUT.append(up)
            INPUT.append(down)
            INPUT.append(left)
            INPUT.append(right)

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

            center_x            = posx + chwdth/2
            center_y            = posy + chhght/2
            d_left_wall         = center_x / 36
            d_right_wall        = width - center_x / 36
            d_top_wall          = center_y / 36
            d_bottom_wall       = (height - center_y) / 36
            d_left_up_wall      = (min(center_x, center_y) * sqrt_2) / 36
            d_left_down_wall    = (min(center_x, height - center_y) * sqrt_2) / 36
            d_right_up_wall     = (min(width - center_x, center_y) * sqrt_2) / 36
            d_right_down_wall   = (min(width - center_x, height - center_y) * sqrt_2) / 36

            INPUT.append(d_top_wall)
            INPUT.append(d_right_up_wall)
            INPUT.append(d_right_wall)
            INPUT.append(d_right_down_wall)
            INPUT.append(d_bottom_wall)
            INPUT.append(d_left_down_wall)
            INPUT.append(d_left_wall)
            INPUT.append(d_left_up_wall)

            for tail in trail:
                temp_dx_pos = []
                temp_dx_neg = []
                temp_dy_pos = []
                temp_dy_neg = []
                if tail[0] == posx:
                    if (posy - tail[1]) > 0:
                        temp_dy_neg.append(posy - tail[1])
                    else:
                        temp_dy_pos.append(posy - tail[1])
                if tail[1] == posy:
                    if (posx - tail[0]) > 0:
                        temp_dx_neg.append(posx - tail[0])
                    else:
                        temp_dx_pos.append(posx - tail[0])


            if temp_dy_neg == []:
                d_tail_up = 0
            else:
                d_tail_up = min(temp_dy_neg) / 36

            if temp_dy_pos == []:
                d_tail_down = 0
            else:
                d_tail_down = max(temp_dy_pos) / 36

            if temp_dx_neg == []:
                d_tail_left = 0
            else:
                d_tail_left = min(temp_dx_neg) / 36

            if temp_dx_pos == []:
                d_tail_right = 0
            else:
                d_tail_right = max(temp_dx_pos) / 36

            INPUT.append(d_tail_up)
            INPUT.append(d_tail_down)
            INPUT.append(d_tail_left)
            INPUT.append(d_tail_right) # input ends
            layer_start = individual.weighted_linear(INPUT, 4)
            layer_end = individual.softmax(layer_start, 4)
            direction = layer_end.index(max(layer_end))
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