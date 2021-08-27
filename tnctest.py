import obj
import numpy as np

model = obj.Sequential([
    obj.ReLU(10,3),
    obj.ReLU(3,4),
    obj.softmax(4)
])

out = model.forward([3 for x in range(10)])
print(out)