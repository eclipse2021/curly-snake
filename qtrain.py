import random
import math
import torch
from torch import nn
import torch.optim as optim

from collections import namedtuple, deque

import net
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


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

    print_screen()
    return x_data
def get_screen():
    global posx, posy, timepassed, x_data, taillen, trail
    #꼬리 좌표 업데이트
    trail.append([posx, posy])                                     # 길이가 하나 늘어남(머리 위치에서)
    last_tail = (int(trail[0][1]/chhght), int(trail[0][0]/chwdth)) #없어질 꼬리 좌표
    if len(trail) - 1 > taillen:
        print("[trail0]", end = '\0')
        print(str(trail[0][0]/chhght) + ' , ' + str(trail[0][1]/chhght))
        del trail[0]                                               # 길이가 하나 줄어듬(맨 마지막 꼬리에서) -> 전체 길이는 일정함
    
    print("[debug]")
    for t in trail:
        print(str(t[0]/chhght) + ',' + str(t[1]/chhght))
    tails_found = 0
    while tails_found < taillen: # 입력 텐서 업데이트
        print("[trailsfound]", end = '\0')
        print(str(trail[tails_found][1]/chhght) + ' , ' + str(trail[tails_found][0]/chhght))
        x_data[0][0][int(trail[tails_found][1]/chhght)][int(trail[tails_found][0]/chwdth)] = 100
        tails_found += 1
    if len(trail) - 1 > taillen:
        print("[0edTensor]", end = '\0')
        print(str(last_tail[1]) + ' , ' + str(last_tail[0]))
    x_data[0][0][last_tail[0]][last_tail[1]] = 0
    x_data[0][0][int(posy/chhght)][int(posx/chwdth)] = 220 # 강한 머리/업데이트 끝

    print_screen()
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

    print("[update]x:" + str(posx/chhght) + "  y:" + str(posy/chwdth))

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
    print("[restart_game()]")
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
num_episodes = 500

x_data = torch.zeros(shape).to(device)
init_screen = get_screen()
_, _, screen_height, screen_width = init_screen.shape

n_actions = 4

policy_net = net.DQN(screen_height, screen_width, n_actions).to(device)
target_net = net.DQN(screen_height, screen_width, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

agent = net.CNN().to(device)


# test_action = torch.tensor([0., 0., 0., 0.]).to(device)
# test_predictions = torch.tensor([0.]).to(device)
# index = 0
# for test in range(4):
#     test_action[test] = 1
#     test_prediction = agent(get_screen(), test_action)
#     print(test_prediction)
#     if test_prediction > test_predictions:
#         test_predictions = test_prediction
#         index = test
# print("final decision", end='\0')
# print(index)
# input("debug stop")

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)

steps_done = 0

def select_action():
    global steps_done
    sample = random.random()

    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    print("[select_action()][eps_threshold]", eps_threshold)
    if sample > eps_threshold:
        print("[select_action()][sample]greater")
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
        return (last_prediction, index)

    else:
        print("[select_action()][sample]less")
        index = random.randrange(n_actions)
        action = torch.tensor([0 for k in range(4)]).to(device)
        action[index] = 1
        return (agent(get_screen_0(), action), index)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). 이것은 batch-array의 Transitions을 Transition의 batch-arrays로
    # 전환합니다.
    batch = Transition(*zip(*transitions))

    # 최종이 아닌 상태의 마스크를 계산하고 배치 요소를 연결합니다
    # (최종 상태는 시뮬레이션이 종료 된 이후의 상태)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                        batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Q(s_t, a) 계산 - 모델이 Q(s_t)를 계산하고, 취한 행동의 열을 선택합니다.
    # 이들은 policy_net에 따라 각 배치 상태에 대해 선택된 행동입니다.
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # 모든 다음 상태를 위한 V(s_{t+1}) 계산
    # non_final_next_states의 행동들에 대한 기대값은 "이전" target_net을 기반으로 계산됩니다.
    # max(1)[0]으로 최고의 보상을 선택하십시오.
    # 이것은 마스크를 기반으로 병합되어 기대 상태 값을 갖거나 상태가 최종인 경우 0을 갖습니다.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # 기대 Q 값 계산
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Huber 손실 계산
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # 모델 최적화
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

for ep in range(num_episodes):
    while True:
        direction = select_action()
        y = update(direction[1])

        if y == 0:
            restart_game()
            break
        else:
            print(y)
        print(direction)
        input("debug stop\n")
        timepassed += 1
        if timepassed % 10 == 0:
            taillen += 1


for ep in range(num_episodes):
    timepassed = 0
    x_data = torch.zeros(shape).to(device)
    while True:    #main loop
        y = torch.tensor([1.]).to(device)
        optimizer.zero_grad()
        predictions = []
        predictions.append(model(x_data, torch.tensor([1.,0.,0.,0.,]).to(device))) #   up
        predictions.append(model(x_data, torch.tensor([0.,1.,0.,0.,]).to(device))) # down
        predictions.append(model(x_data, torch.tensor([0.,0.,1.,0.,]).to(device))) # left
        predictions.append(model(x_data, torch.tensor([0.,0.,0.,1.,]).to(device))) #right
        prediction = max(predictions)
        direction = predictions.index(max(predictions)) # 예측
        if direction == 0: # 플레이어 위치 업데이트
            posy -= chhght #
        if direction == 1: #
            posy += chhght #
        if direction == 2: #
            posx -= chhght #
        if direction == 3: #
            posx += chhght #

        trail.append([posx, posy])                                     # 길이가 하나 늘어남(머리 위치에서)
        last_tail = (int(trail[0][1]/chhght), int(trail[0][0]/chwdth)) #없어질 꼬리 좌표
        if len(trail) - 1 > taillen:
            del trail[0]                                               # 길이가 하나 줄어듬(맨 마지막 꼬리에서) -> 전체 길이는 일정함

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

        cost = criterion(prediction, y)
        cost.backward()
        optimizer.step()

        if y == torch.tensor([0.]).to(device):
            if ep % 100 == 0:
                print("ep" + str(ep) + ':' + "survived" + str(timepassed) + "moves|cost=" + str(cost.item()))
            break

        tails_found = 0
        while tails_found < taillen: # 입력 텐서 업데이트
            x_data[0][0][int(trail[tails_found][1]/chhght)][int(trail[tails_found][0]/chwdth)] = 100
            tails_found += 1
        if len(trail) > taillen:
            x_data[0][0][last_tail[0]][last_tail[1]] = 0
        x_data[0][0][int(posx/chhght)][int(posy/chwdth)] = 220



        timepassed += 1
        if timepassed % 10 == 0:
            taillen += 1
    trail               = []
    taillen             = 1
    posx                = (width/2) - chwdth
    posy                = height/2 - chhght
