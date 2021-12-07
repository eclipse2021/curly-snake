import random
import math
import torch
from torch import nn
import torch.optim as optim

from collections import namedtuple, deque
import copy

import net

Transition = namedtuple('Transition', ('state', 'action', 'reward'))

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#game settings -----------------------------
width              = 720    #가로20칸
height             = 720    #세로20칸
shape = (1, 1, 20, 20)      #screen size

chwdth             = 36
chhght             = chwdth
posx               = (width/2) - chwdth
posy               = height/2 - chhght

trail              = []
direction          = random.randrange(0,3)
timepassed         = 0
taillen            = 1
#--------------------------------------

def get_screen_0():
    global posx, posy, timepassed, x_data, taillen, trail
    trail.append([posx, posy])
    if len(trail) > taillen:
        del trail[0]
    
    x_data = torch.zeros(shape).to(device)
    for t in trail:
        x_data[0][0][int(t[1]/chhght)][int(t[0]/chhght)] = 100
    x_data[0][0][int(posy/chhght)][int(posx/chwdth)] = 220

    #print_screen()
    return x_data

def update(_input):
    global posx, posy, timepassed, x_data
    y = torch.tensor([1.]).to(device)
    if _input == 0:
        posy -= chhght 
    elif _input == 1: 
        posy += chhght
    elif _input == 2:
        posx -= chhght
    elif _input == 3:
        posx += chhght

    #print("[update]x:" + str(posx/chhght) + "  y:" + str(posy/chwdth))

    if posx < 0:                              # 사망판정 시작
        y = torch.tensor([0.]).to(device)
    if posx > width - chwdth:
        y = torch.tensor([0.]).to(device)
    if posy < 0:
        y = torch.tensor([0.]).to(device)
    if posy > height - chhght:
        y = torch.tensor([0.]).to(device)
    
    if len(trail) > 1:
        if [posx, posy] in trail[:len(trail) - 1]:
            y = torch.tensor([0.]).to(device) # 사망판정 끝
    return y

def print_screen():
    global x_data
    game_screen = x_data.tolist()
    for u in game_screen[0][0]:
        for v in u:
            if v == 0:
                print('0 ', end = '\0')
            if v == 100:
                print('1 ', end = '\0')
            if v == 220:
                print('H ', end = '\0')
        print()

def restart_game():
    global x_data, trail, taillen, posx, posy
    #print("[restart_game()]")
    x_data = torch.zeros(shape).to(device)
    trail               = []
    taillen             = 1
    posx                = (width/2) - chwdth
    posy                = height/2 - chhght

# hyperparameters
BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
num_episodes = 50000

x_data = torch.zeros(shape).to(device)

n_actions = 4

target_network = net.CNN().to(device)
agent = net.CNN().to(device)

optimizer = optim.Adam(target_network.parameters())
memory = ReplayMemory(10000)

steps_done = 0

def select_action():
    global steps_done
    sample = random.random()

    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    current_screen = get_screen_0()
    #print("[select_action()][eps_threshold]", eps_threshold)
    if sample > eps_threshold:
        #print("[select_action()][sample]greater")
        last_prediction = torch.tensor([0.]).to(device)
        action = torch.tensor([0 for k in range(4)]).to(device)
        index = 0
        current_screen = get_screen_0()
        for x in range(4):
            action[x] = 1
            action[x - 1] = 0
            prediction = agent(current_screen, action)
            if prediction > last_prediction:
                index = x
                last_prediction = prediction
        return (current_screen, last_prediction, index)

    else:
        #print("[select_action()][sample]less")
        index = random.randrange(n_actions)
        action = torch.tensor([0 for k in range(4)]).to(device)
        action[index] = 1
        return (current_screen, agent(current_screen, action), index)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        print("gathering memory...('^')({})".format(len(memory)))
        return
    batch = memory.sample(BATCH_SIZE)

    criterion = torch.nn.BCELoss().to(device)
    avg_loss = 0
    action = torch.tensor([0 for k in range(4)]).to(device)
    for data in batch:
        optimizer.zero_grad()
        action[data[1]] = 1
        prediction = target_network(data[0], action)
        loss = criterion(prediction, data[2])
        loss.backward()
        optimizer.step()
        avg_loss += loss / BATCH_SIZE
    return avg_loss

for ep in range(num_episodes):
    frames_survived = 0
    last_loss = 0
    while True:
        direction = select_action()
        y = update(direction[2])
        memory.push(copy.copy(direction[0]), copy.copy(direction[2]), copy.copy(y))
    
        last_loss = optimize_model()

        if y == 0:
            restart_game()
            break
        frames_survived += 1
    if ep % TARGET_UPDATE == 0:
        agent.load_state_dict(target_network.state_dict())
    if ep % 100 == 0:
        print("episode{}...".format(ep), end = '\0')
        print("survived {} frames ...".format(frames_survived), end = '\0')
        print("cost = {}".format(last_loss))

