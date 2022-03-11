import random
import numpy as np
import csv
from MultiCompanyAgent import IAC
from MultiCompanyEnv import env
from MultiCompanyNormalizer import Normalizer
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import tensorflow as tf
import time
import random

print("Major Revision Experiment Start")

save_path = "./MajorRevision"
train_mode = True
# Load model은 Agent_discre.py에서도 변경해줘야함
load_model = False

# Number of Agents
num_agents = 7
env = env(num_agents, load_model)
state_size = env.state_size
action_size = env.action_size

total_agents = num_agents
EP_MAX = 30000
EP_LEN = env.MaxTime
BATCH = env.batch_size

A_LR = 0.00002
C_LR = 0.0001

# Path of env
epsilon = 1.0
MaxEpsilon = 0.6
MinEpsilon = 0.001

net = IAC(save_path)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


normalizer = Normalizer(state_size)

if train_mode:
    for ep in range(EP_MAX):
        #ra=int((time.time()-round(time.time())+0.5)*3000)
        ra = np.random.randint(363)
        rb = np.random.randint(5)
        state = env.reset(ra+1,rb)
        ep_r1 = 0
        ep_r2 = 0
        ep_r3 = 0
        ep_r4 = 0
        ep_r5 = 0
        ep_r6 = 0
        ep_r7 = 0

        buffer_s, buffer_s_next, buffer_a,  buffer_r, buffer_t , buffer_prob= [], [], [], [], [] ,[]

        for t in range(EP_LEN):

            normalizer.observe(state[0])
            normalizer.observe(state[1])
            normalizer.observe(state[2])
            normalizer.observe(state[3])
            normalizer.observe(state[4])
            normalizer.observe(state[5])
            normalizer.observe(state[6])

            state = normalizer.normalize(state)
            state1 = state[0]
            state2 = state[1]
            state3 = state[2]
            state4 = state[3]
            state5 = state[4]
            state6 = state[5]
            state7 = state[6]

            state = np.reshape(np.transpose(state), newshape=(-1, state_size, num_agents))
            actions = net.action_prob(state, False)
            actions = actions[0]

            actions = np.transpose(actions)
            x =  np.reshape(actions, newshape=( num_agents, action_size))
            x1 = x[0]
            x2 = x[1]
            x3 = x[2]
            x4 = x[3]
            x5 = x[4]
            x6 = x[5]
            x7 = x[6]


            a1 = np.random.choice(action_size, 1,  p=x1)
            a2 = np.random.choice(action_size, 1,  p=x2)
            a3 = np.random.choice(action_size, 1,  p=x3)
            a4 = np.random.choice(action_size, 1,  p=x4)
            a5 = np.random.choice(action_size, 1,  p=x5)
            a6 = np.random.choice(action_size, 1,  p=x6)
            a7 = np.random.choice(action_size, 1,  p=x7)

            prob_a1 = x1[a1].item()
            prob_a2 = x2[a2].item()
            prob_a3 = x3[a3].item()
            prob_a4 = x4[a4].item()
            prob_a5 = x5[a5].item()
            prob_a6 = x6[a6].item()
            prob_a7 = x7[a7].item()


            acts1 = a1 * 0.002 + 0.98        #tap1         1.00 ~ 1.04 
            acts2 = a2 * 0.1   - 1           #sop1(69-15) -1 ~ 1
            acts3 = a3 * 0.03  - 0.3         #shunt69 -0.3 ~ 0.3
            acts4 = a4 * 0.03  - 0.3         #shunt15 -0.3 ~ 0.3
            acts5 = a5 * 0.1   - 1           #sop2(27-54) -1 ~ 1
            acts6 = a6 * 0.03  - 0.3         #shunt69 -0.3 ~ 0.3
            acts7 = a7 * 0.03  - 0.3         #shunt69 -0.3 ~ 0.3

            print(acts1,acts2,acts3,acts4,acts5,acts6,acts7)

            next_states, rewards, terminals, info, infov, infoPV = env.step(acts1,acts2, acts3,acts4,acts5,acts6,acts7)
            '''
            with open('./MajorRevision/info1.csv', 'a', newline='') as mycsvfile:
                wr = csv.writer(mycsvfile)
                 
                wr.writerow(info)
            with open('./MajorRevision/infos1.csv', 'a', newline='') as mycsvfile:
                wr = csv.writer(mycsvfile)
                 
                wr.writerow(infos)    
            with open('./MajorRevision/infov1.csv', 'a', newline='') as mycsvfile:
                wr = csv.writer(mycsvfile)
                 
                wr.writerow(infov)        
            '''
            ep_r1 += rewards[0]
            ep_r2 += rewards[1]
            ep_r3 += rewards[2]
            ep_r4 += rewards[3]
            ep_r5 += rewards[4]
            ep_r6 += rewards[5]
            ep_r7 += rewards[6]

            # Reconfiguration for action
            tmp = np.zeros(shape=(action_size, num_agents))
            tmp[a1[0]][0] = 1
            tmp[a2[0]][1] = 1
            tmp[a3[0]][2] = 1
            tmp[a4[0]][3] = 1
            tmp[a5[0]][4] = 1
            tmp[a6[0]][5] = 1
            tmp[a7[0]][6] = 1

            state_next = np.copy(next_states)
            state_next = normalizer.normalize(state_next)
            state_next1 = state_next[0]
            state_next2 = state_next[1]
            state_next3 = state_next[2]
            state_next4 = state_next[3]
            state_next5 = state_next[4]
            state_next6 = state_next[5]
            state_next7 = state_next[6]
            state_next = np.reshape(np.transpose(state_next), newshape=(-1, state_size, num_agents))
            proba = np.zeros(shape=(action_size, num_agents))

            proba[a1[0]][0] = prob_a1
            proba[a2[0]][1] = prob_a2
            proba[a3[0]][2] = prob_a3
            proba[a4[0]][3] = prob_a4
            proba[a5[0]][4] = prob_a5
            proba[a6[0]][5] = prob_a6
            proba[a7[0]][6] = prob_a7
            # Store transition ==================
            buffer_s.append(state)
            buffer_s_next.append(state_next)
            buffer_a.append(tmp)
            buffer_r.append(rewards)
            buffer_t.append(terminals)
            buffer_prob.append(proba)

            # ===================================

            state = np.copy(next_states)

            #print("[actions]", actions)
            #print("[acts]", acts)
            # print("[temp ac]", tmp)
            # print(rewards)

            # Train network ================================================================================================
            if (t + 1) % 4 == 0 or t == EP_LEN - 1 and train_mode:
                # if train_mode and t == 1:
                bs = np.reshape(buffer_s, newshape=(-1, state_size, num_agents))
                bs_ = np.reshape(buffer_s_next, newshape=(-1, state_size, num_agents))
                ba = np.reshape(buffer_a, newshape=(-1, action_size, num_agents))
                br = np.reshape(buffer_r, newshape=(-1, num_agents))
                bt = np.reshape(buffer_t, newshape=(-1, num_agents))
                #print('bt=',bt)
                bprob = np.reshape(buffer_prob, newshape=(-1, action_size, num_agents))

                for i in range(3):
                    loss = net.train_op(s=bs, s_next=bs_, a=ba, r=br, p =bprob , t=bt, c_lr=C_LR, a_lr=A_LR)
                    #print('i' , i)
                buffer_s, buffer_s_next, buffer_a, buffer_r, buffer_t ,buffer_prob = [], [], [], [], [], []
            # ==============================================================================================================

        if (ep + 1) % 10 == 0:
            net.save_model(save_path)
            # self.n = np.zeros(nb_inputs)
            # self.mean = np.zeros(nb_inputs)
            # self.mean_diff = np.zeros(nb_inputs)
            # self.var = np.zeros(nb_inputs)
            weights = [normalizer.n, normalizer.mean, normalizer.mean_diff, normalizer.var]
            np.save('./MajorRevision/normalizer_weight', weights)

        print("[Ep] {} [Reward] {} [A_Loss] {} [C_Loss] {} [Epsilon] {}".format(ep+1, ep_r1 + ep_r2 + ep_r3 + ep_r4+ ep_r5 + ep_r6 + ep_r7,  loss[0], loss[1], epsilon))
        if train_mode:
            with open('./MajorRevision/reward1.csv', 'a', newline='') as mycsvfile:
                wr = csv.writer(mycsvfile)
                tmp = ep_r1 + ep_r2 + ep_r3 + ep_r4+ ep_r5 + ep_r6 + ep_r7
                wr.writerow([ep_r1,ep_r2,ep_r3,ep_r4,ep_r5,ep_r6,ep_r7, tmp])
                
else:
    normalizer_weights = np.load(save_path + '/normalizer_weight.npy')
    # print(normalizer_weights)
    normalizer.n = normalizer_weights[0]
    normalizer.mean = normalizer_weights[1]
    normalizer.mean_diff = normalizer_weights[2]
    normalizer.var = normalizer_weights[3]

    state = env.reset()
    ep_r1 = 0
    ep_r2 = 0
    for t in range(EP_LEN):
        normalizer.observe(state[0])    
        normalizer.observe(state[1])    

        state = normalizer.normalize(state)
        state = np.reshape(np.transpose(state), newshape=(-1, state_size, num_agents))
        actions = net.action_prob(state, False)

        actions = np.reshape(actions, newshape=(action_size, num_agents))
        acts = np.argmax(actions, axis=0)

        if acts[0]==0:
            acts1 = -0.5
        elif acts[0]==1:
            acts1 = -0.4
        elif acts[0]==2:
            acts1 = -0.3
        elif acts[0]==3:
            acts1 = -0.2
        elif acts[0]==4:
            acts1 = 0.2
        elif acts[0]==5:
            acts1 = 0.3
        elif acts[0]==6:
            acts1 = 0.4
        elif acts[0]==7:
            acts1 = 0.5

        if 5*soc1 + acts1 > 4:
            acts1 = -0.5               

        elif 5*soc1 + acts1 < 1:
            acts1 = 0.5
        else :
            acts1 = acts1

        if acts[1]==0:
            acts2 = -0.5
        elif acts[1]==1:
            acts2 = -0.4
        elif acts[1]==2:
            acts2 = -0.3
        elif acts[1]==3:
            acts2 = -0.2
        elif acts[1]==4:
            acts2 = 0.2
        elif acts[1]==5:
            acts2 = 0.3
        elif acts[1]==6:
            acts2 = 0.4
        elif acts[1]==7:
            acts2 = 0.5

        if 2.5*soc2 + acts2 > 2:
            acts2 = -0.5
        elif 2.5*soc2 + acts2 < 0.5:
            acts2 = 0.5
        else :
            acts2 = acts2

       
        next_states, rewards, terminals, info, infov = env.step(acts1,acts2, soc1,soc2)

        ep_r1 += rewards[0]
        ep_r2 += rewards[1]

        # Reconfiguration for action
        tmp = np.zeros(shape=(action_size, num_agents))


        if acts1 == -0.5 :
            acts1 = 0
        elif acts1 == -0.4:
            acts1 = 1
        elif acts1 == -0.3:
            acts1 = 2
        elif acts1 == -0.2:
            acts1 = 3
        elif acts1 == 0.2:
            acts1 = 4
        elif acts1 == 0.3:
            acts1 = 5
        elif acts1 == 0.4:
            acts1 = 6
        elif acts1 == 0.5:
            acts1 = 7

        if acts2 == -0.5 :
            acts2 = 0
        elif acts2 == -0.4:
            acts2 = 1
        elif acts2 == -0.3:
            acts2 = 2
        elif acts2 == -0.2:
            acts2 = 3
        elif acts2 == 0.2:
            acts2 = 4
        elif acts2 == 0.3:
            acts2 = 5
        elif acts2 == 0.4:
            acts2 = 6
        elif acts2 == 0.5:
            acts2 = 7
        tmp[acts[0]][0] = 1
        tmp[acts[1]][1] = 1

        print("[Info] {} [Actions] {}".format(info, acts))
       
        state = np.copy(next_states)
        with open('./MajorRevision/check1.csv', 'a', newline='') as mycsvfile:
            wr = csv.writer(mycsvfile)             
            wr.writerow(info)
        with open('./MajorRevision/checkv1.csv', 'a', newline='') as mycsvfile:
            wr = csv.writer(mycsvfile)             
            wr.writerow(infov)

    print(ep_r1, ep_r2)
