import numpy as np
import decimal
import random

from numpy.lib.function_base import percentile

constants_e = 2.71828
c_e = str(np.exp(1))

class Sequential:
    def __init__(self, sequence: list) -> None:
        self.sequence = sequence
    
    def forward(self, IN):
        for layer in self.sequence:
            IN = layer.activate(IN)
        return IN

class ReLU:
    def __init__(self, in_features, out_features) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.w = [[0 for x in range(in_features)] for y in range(out_features)]
        self.b = [0 for x in range(out_features)]
        for index_y in range(out_features):
            for index_x in range(self.in_features):
                self.w[index_y][index_x] = random.uniform(-1, 1)
        for index_x in range(out_features):
            self.b[index_x] = random.uniform(-1, 1)

    
    def relu(slef, x):
        return max(0,x)

    def activate(self, IN: list) -> list:
        OUT = [0 for x in  range(self.out_features)]
        neurons_activated = 0
        while neurons_activated < self.out_features:
            temp_neuron = 0
            # temp_neuron += (IN[index] * self.w[neurons_activated][index] for index in range(self.in_features))
            # temp_neuron += self.b[neurons_activated]

            for index in range(self.in_features):
                temp_neuron += IN[index] * self.w[neurons_activated][index]
            temp_neuron += self.b[neurons_activated]
            OUT[neurons_activated] = self.relu(temp_neuron)
            neurons_activated += 1
        return OUT

class softmax:
    def __init__(self, dimensions) -> None:
        self.dimensions = dimensions
    def activate(self, IN: list) -> list:
        c = max(IN)
        IN = [x - c for x in IN]
        sum = 0
        for x in IN:
            sum += np.exp(x)
        OUT = [np.exp(x)/sum for x in IN]
        return OUT




class Individual:
    # input_nodes = 400
    # hidden_nodes_1 = 16
    # hidden_nodes_2 = 16
    # output_nodes = 4
    def __init__(self) -> None: # fira code
        self.w = [[[0 for x in range(400)] for y in range(16)],[[0 for x in range(16)] for y in range(16)],[[0 for x in range(16)] for y in range(4)]]
        self.b = [[0 for x in range(16)], [0 for x in range(16)], [0 for x in range(4)]]
        self.fit = 0
        for i in range(0, 16):
            for j in range(0, 400):
                self.w[0][i][j] = random.random() * random.choice([1, -1])
        for i in range(0, 16):
            for j in range(0, 16):
                self.w[1][i][j] = random.random() * random.choice([1, -1])
        for i in range(0, 4):
            for j in range(0, 16):
                self.w[2][i][j] = random.random() * random.choice([1, -1])
        

        for i in range(0, 16):
            self.b[0][i] = random.random() * random.choice([1, -1])
        for i in range(0, 16):
            self.b[1][i] = random.random() * random.choice([1, -1])
        for i in range(0, 4):
            self.b[2][i] = random.random() * random.choice([1, -1])

    def __init__legacy(self):
        self.w = []
        self.b = []
        self.fitness = 0
        for _ in range(0, 800):
            self.w.append(random.random() * random.choice([1, -1]) *0.1)
        for _ in range(0, 44):
            self.b.append(random.random() * random.choice([1, -1]) *0.1)

    def forward(self, IN, depth, hid_layer_nodes):    #(Individual, list, int, list[int], int)
        '''
        !legacy function
        '''
        inputlayer = []    #상대 입력 레이어: 지금 깊이에서의 입력층
        inputlayer = IN    #초기화
        outputlayer = []    #상대 출력 레이어: 지금 깊이에서의 출력층
        node_temp = 0    #노드의 값을 저장
        w_step_hint = 0    #개체의 w값 위치를 찾는데 도움(깊이가 늘어날수록 지나온 스탭만큼 쌓인다)
        b_step_hint = 0    #개체의 b값 위치를 찾는데 도움(깊이가 늘어날수록 지나온 스탭만큼 쌓인다)
        for forward_layer in range(0, depth):    #깊이
            for output_nodes in range(0, hid_layer_nodes[forward_layer]):    #상대 출력 노드를 차례로 가리킴
                node_temp = 0
                for input_nodes in range(0, len(inputlayer)):    #상대 입력 노드를 차례로 가리킴
                    # if forward_layer < 2:
                    #     node_temp += leaky_Relu(inputlayer[input_nodes])
                    # else:
                    #     node_temp += float(round(sigmoid_custom(inputlayer[input_nodes] * self.w[w_step_hint + input_nodes])))
                    
                    node_temp += inputlayer[input_nodes] * self.w[w_step_hint + input_nodes]
                    # if forward_layer < 2:
                    #     node_temp += leaky_Relu(inputlayer[input_nodes])
                    # else:
                    #     node_temp += sigmoid(inputlayer[input_nodes] * self.w[w_step_hint + input_nodes])
                node_temp += self.b[b_step_hint + output_nodes]    #출력 노드 하나에 대한 forward연산이 끝남
                node_temp = float(round(sigmoid_custom(node_temp)))
                outputlayer.append(node_temp)
                w_step_hint += len(inputlayer)
            b_step_hint += hid_layer_nodes[forward_layer]

            inputlayer = outputlayer
            outputlayer = []

        return inputlayer # 위의 for문이 다 끝나면 최종 출력층이 inputlayer에 옮겨짐

    def softmax(self, IN, dimention):
        p = []
        IN_sum = decimal.Decimal('0')
        for i in range(0, dimention):
            IN_sum += decimal.Decimal(c_e) ** decimal.Decimal(str(IN[i]))
        for i in range(0, dimention):
            p.append(float(decimal.Decimal(c_e) ** decimal.Decimal(str(IN[i]))/IN_sum))

        return p

    def weighted_linear(self, localIN: list, output_dimension):
        OUT = [0 for i in range(output_dimension)]
        for i in range(0, len(localIN)):
            for j in range(0, output_dimension):
                OUT[j] += localIN[i] * self.w[i][j]
        
        for i in range(0, output_dimension):
            OUT[i] += self.b[i]
        
        return OUT

def step_function(x):
    if x>0:
        return 1
    else:
        return -1

def sigmoid_decimal(x):
    ex = 1 + np.exp(-x)
    return decimal.Decimal(1) / decimal.Decimal(ex)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_custom(x):#
    res = (2.71828 ** x) + 1
    return decimal.Decimal(1) / decimal.Decimal(res)

def identity(x):
    return x

def Relu(x):
    return max(0, x)

def leaky_Relu(x):
    if x < 0:
        return 0.01 * x
    else:
        return max(0, x)

def crs(prnt_1, prnt_2):
    '''
    이것은 두 배열을 포인트 한개 기준으로 뒤집기만 하는
    (대상1[리스트], 대상2[리스트])
    '''
    crossover_point = random.randrange(0, len(prnt_1) - 1)
    desc_I = []
    desc_II = []
    for _ in range(0, crossover_point):
        desc_I.append(prnt_1[_])
    for _ in range(0, len(prnt_1) - crossover_point):
        desc_I.append(prnt_2[_ + crossover_point])

    for _ in range(0, crossover_point):
        desc_II.append(prnt_2[_])
    for _ in range(0, len(prnt_1) - crossover_point):
        desc_II.append(prnt_1[_ + crossover_point])
    return desc_I, desc_II

def crs_mean(parent1, parent2):
    '''
    두 배열의 평균과 0.3만큼 치우친 배열을 반환하는
    '''
    temp_mean = []
    temp_3 = []
    for pointer in range(len(parent1)):
        temp_mean.append((parent1[pointer] + parent2[pointer])/2)
    
    for pointer in range(len(parent1)):
        temp_3.append((parent1[pointer] + parent2[pointer])/3)

    return temp_mean, temp_3

def crossover_operator(crossover_points, prnt_1, prnt_2):
    '''
    이것은 crs를 여러번 돌릴수 있게 하는
    
    (크로스오버 포인트 수[정수], 대상1[리스트], 대상2[리스트])
    '''
    container = (prnt_1, prnt_2)
    for _ in range(0, crossover_points):
        out_ = crs(container[0], container[1])
        container = (out_[0], out_[1])
    return out_

def crossover_active(parent_grp_pointed):
    temp = []
    temp = parent_grp_pointed
    out_grp = []
    for point in range(0,int(len(parent_grp_pointed)/2)):
        parent_x = random.choice(parent_grp_pointed)
        parent_grp_pointed.remove(parent_x)
        parent_y = random.choice(parent_grp_pointed)
        parent_grp_pointed.remove(parent_y)
        
        temp_w = crossover_operator(random.randint(1, 10), parent_x.w, parent_y.w)
        temp_b = crossover_operator(random.randint(1, 10), parent_x.b, parent_y.b)[1]

        parent_x.w = temp_w[0]
        parent_y.w = temp_w[1]
        parent_x.b = temp_b[0]
        parent_y.b = temp_b[1]
        out_grp.append(parent_x)
        out_grp.append(parent_y)
    '''
    for point in range(0,int(len(parent_grp)/2)):
        parent_x = random.choice(parent_grp)
        parent_grp.remove(parent_x)
        parent_y = random.choice(parent_grp)
        parent_grp.remove(parent_y)

        parent_x.w = crs_mean(parent_x.w, parent_y.w)[0]
        parent_x.b = crs_mean(parent_x.b, parent_y.b)[0]
        parent_y.w = crs_mean(parent_x.w, parent_y.w)[1]
        parent_y.b = crs_mean(parent_x.b, parent_y.b)[1]
        out_grp.append(parent_x)
        out_grp.append(parent_y)
        '''
    return out_grp

def crossover_mean_active(parent_grp):
    out_grp = []
    for point in range(0,int(len(parent_grp)/2)):
        parent_x = random.choice(parent_grp)
        parent_grp.remove(parent_x)
        parent_y = random.choice(parent_grp)
        parent_grp.remove(parent_y)

        parent_x.w = crs_mean(parent_x.w, parent_y.w)[0]
        parent_x.b = crs_mean(parent_x.b, parent_y.b)[0]
        parent_y.w = crs_mean(parent_x.w, parent_y.w)[1]
        parent_y.b = crs_mean(parent_x.b, parent_y.b)[1]
        out_grp.append(parent_x)
        out_grp.append(parent_y)

    return out_grp


def crossover(group):
    '''
    [크로스오버 함수]
    -50% 평균
    -50% 랜덤포인트
    크로스오버를 구현한
    *group은 인구모델을 받음
    '''
    input_group = group
    output_group = []

    for _ in range(0, int(len(input_group)/2)):
        target1 = random.choice(group)
        group.remove(target1)
        target2 = random.choice(group)
        group.remove(target2)

        randpoint = random.randint(2, 10)
        temp_w = crossover_operator(randpoint, target1.w, target2.w)
        temp_b = crossover_operator(randpoint, target1.b, target2.b)
        target1.w = temp_w[0]
        target2.w = temp_w[1]
        target1.b = temp_b[0]
        target2.b = temp_b[1]
        output_group.append(target1)
        output_group.append(target2)

    for _ in range(0, int(len(input_group)/2)):
        target1 = random.choice(group)
        group.remove(target1)
        target2 = random.choice(group)
        group.remove(target2)

        temp_w = crs_mean(target1.w, target2.w)
        temp_b = crs_mean(target1.b, target2.b)

        target1.w = temp_w[0]
        target1.b = temp_b[0]
        target2.w = temp_w[1]
        target2.b = temp_b[1]

        output_group.append(target1)
        output_group.append(target2)

    return output_group

def crossover_reproductive(parents_group, total_size: int = 1000, similarity_threshold = 0.01):
    print("\nstarted crossing...") #  디버그 로그
    pending_list = []
    parent_size = len(parents_group)

    while len(parents_group) != 0:
        targetA = random.choice(parents_group)
        parents_group.remove(targetA)
        targetB = random.choice(parents_group)
        parents_group.remove(targetB)

        for point in range(int(total_size/parent_size)):
            #print("그룹:{}".format(point)) # 디버그 로그
            a = targetA
            b = targetB
            pending_list.append(a)
            pending_list.append(b)

            print("[g{0}]targetA:{1}\ttargetB:{2}".format(point, pending_list[-2].w[:5], pending_list[-1].w[:5]))

            pointer = 0
            while pointer < len(pending_list[-2].w):
                #print("w crossing...")
                if abs(pending_list[-2].w[pointer] - pending_list[-1].w[pointer]) < similarity_threshold:
                    pass
                else:
                    if random.randrange(2) == 1:
                        temp_N2 = pending_list[-2].w[pointer]
                        temp_N1 = pending_list[-1].w[pointer]
                        pending_list[-2].w[pointer] = temp_N1
                        pending_list[-1].w[pointer] = temp_N2
                    else:
                        pass
                pointer += 1
            
            pointer = 0
            while pointer < len(pending_list[-2].b):
                #print("b crossing...")
                if abs(pending_list[-2].b[pointer] - pending_list[-1].b[pointer]) < similarity_threshold:
                    pass
                else:
                    if random.randrange(2) == 1:
                        temp_N2 = pending_list[-2].b[pointer]
                        temp_N1 = pending_list[-1].b[pointer]
                        pending_list[-2].b[pointer] = temp_N1
                        pending_list[-1].b[pointer] = temp_N2
                pointer += 1

            print("[g{0}]resultA:{1}\tresultB:{2}(nowpending:{3}results)".format(point, pending_list[-2].w[:5], pending_list[-1].w[:5], len(pending_list)))
    
    print("...crossing finished")
    return pending_list

def TNC(IND: list, similarity_threshold = 0.01, mutant_possibility = 0.005):
    random.shuffle(IND)
    for i in range(int(len(IND)/2)):
        left  = IND.pop(0)
        right = IND.pop(0)

        for j in range(16):
            for k in range(4):
                if abs(left.w[j][k] - right.w[j][k]) < similarity_threshold and random.random() < 0.7:
                    left_ikth     =  left.w[j][k]
                    right_ikth    = right.w[j][k]
                    left.w[j][k]  =    right_ikth
                    right.w[j][k] =     left_ikth
        
        if random.random() <= mutant_possibility:
            left.w[random.randint(0, len(left.w) -1)][random.randint(0, len(left.w[0]) -1)]    = random.random()
            right.w[random.randint(0, len(right.w) -1)][random.randint(0, len(right.w[0]) -1)] = random.random()

        IND.append(left)
        IND.append(right)

    for i in range(int(len(IND)/2)):
        left  = IND.pop(0)
        right = IND.pop(0)

        for j in range(4):
            if abs(left.b[j] - right.b[j]) < similarity_threshold and random.random() < 0.7:
                left_element  =     left.b[j]
                right_element =    right.b[j]
                left.b[j]     = right_element
                right.b[j]    =  left_element
        
        if random.random() <= mutant_possibility:
            left.b[random.randint(0, len(left.b) -1)]   = random.random()
            right.b[random.randint(0, len(right.b) -1)] = random.random()

        IND.append(left)
        IND.append(right)


def TNC_legacy(parents: list):    # test : passed
    random.shuffle(parents)
    for i in range(int(len(parents)/2)):
        left  = parents.pop(0)
        right = parents.pop(0)

        for i in range(len(right)):
            if left[i] != right[i] and random.random() < 0.7:
                left_ith  =   left[i]
                right_ith =  right[i]
                left[i]   = right_ith
                right[i]  =  left_ith

        parents.append(left)
        parents.append(right)

    return parents


if __name__ == "__main__":
    model = Individual()
    print(model.b)