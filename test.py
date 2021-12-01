x = [0 for k in range(4)]
for index in range(4):
    x[index] = 1
    x[index - 1] = 0
    print(x)