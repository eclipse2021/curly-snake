from collections import namedtuple, deque
import random

Transition = namedtuple('Transition', ('action', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        #"""transition 저장"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

memory = ReplayMemory(100)

idx = 0
while idx < 10:
    prediction = random.random()
    reward = random.random()
    memory.push(prediction, reward)
    idx += 1

def smp(size):
    return memory.sample(size)

print("initial memory>")
print(memory)
while True:
    batch = smp(5)
    for data in batch:
        print(data)
    input()
