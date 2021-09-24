import obj


population = []
population_size = 4

for i in range(population_size):
    population.append(obj.Sequential([obj.ReLU(10,3)]))

for i in range(population_size):
    print(population[i].sequence[0].w[0])


parents = population[:int(population_size/2)]

off1 = obj.MPX(parents)
off2 = obj.MPX(parents)

parents = off1 + off2

print("\n\n\n")
for i in range(population_size):
    print(population[i].sequence[0].w[0])