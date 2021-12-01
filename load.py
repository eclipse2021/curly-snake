import pygad
import pygad.torchga
filename = input("filename:")
loaded_instance = pygad.load(filename = filename)
global_bestfit = 0
def callback_generation(ga_instance):
    global agent, global_bestfit
    print("Generation" + str(ga_instance.generations_completed), end = '|')
    print("Fitness=" + str(ga_instance.best_solution()[1]))
    if ga_instance.best_solution()[1] > global_bestfit:
        global_bestfit = ga_instance.best_solution()[1]
        filename = str(ga_instance.best_solution()[1]) + 'f'
        ga_instance.save(filename = filename)

loaded_instance.run()
