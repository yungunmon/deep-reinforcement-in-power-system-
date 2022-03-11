import random
import numpy as np
import csv
from MultiCompanyEnv import env
from MultiCompanyNormalizer import Normalizer
from torch.autograd import Variable
from Network import PPO
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import time
import os


print("Major Revision Experiment Start")

save_path = "./MajorRevision"
train_mode = False
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
#MaxEpsilon = 0.6
MinEpsilon = 0.001
print_interval = 10


net1 = PPO(1,save_path)
net2 = PPO(2,save_path)
net3 = PPO(3,save_path)
net4 = PPO(4,save_path)
net5 = PPO(5,save_path)
net6 = PPO(6,save_path)
net7 = PPO(7,save_path)
net1.load_state_dict(torch.load(os.path.join(save_path,'net1_params.pkl')))
net2.load_state_dict(torch.load(os.path.join(save_path,'net2_params.pkl')))
net3.load_state_dict(torch.load(os.path.join(save_path,'net3_params.pkl')))
net4.load_state_dict(torch.load(os.path.join(save_path,'net4_params.pkl')))
net5.load_state_dict(torch.load(os.path.join(save_path,'net5_params.pkl')))
net6.load_state_dict(torch.load(os.path.join(save_path,'net6_params.pkl')))
net7.load_state_dict(torch.load(os.path.join(save_path,'net7_params.pkl')))
normalizer = Normalizer(state_size)

if train_mode:  

    for ep in range(EP_MAX):
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
            prob1 = net1.pi(torch.from_numpy(state1).float())
            prob2 = net2.pi(torch.from_numpy(state2).float())
            prob3 = net3.pi(torch.from_numpy(state3).float())
            prob4 = net4.pi(torch.from_numpy(state4).float())
            prob5 = net5.pi(torch.from_numpy(state5).float())
            prob6 = net6.pi(torch.from_numpy(state6).float())
            prob7 = net7.pi(torch.from_numpy(state7).float())

            m1 = Categorical(prob1)
            m2 = Categorical(prob2)
            m3 = Categorical(prob3)
            m4 = Categorical(prob4)
            m5 = Categorical(prob5)
            m6 = Categorical(prob6)
            m7 = Categorical(prob7)


            a1 = m1.sample().item()
            a2 = m2.sample().item()
            a3 = m3.sample().item()
            a4 = m4.sample().item()
            a5 = m5.sample().item()
            a6 = m6.sample().item()
            a7 = m7.sample().item()

            prob_a1 = prob1[a1].item()
            prob_a2 = prob2[a2].item()
            prob_a3 = prob3[a3].item()
            prob_a4 = prob4[a4].item()
            prob_a5 = prob5[a5].item()
            prob_a6 = prob6[a6].item()
            prob_a7 = prob7[a7].item()

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

            with open('./MajorRevision/info2.csv', 'a', newline='') as mycsvfile:
                wr = csv.writer(mycsvfile)
                wr.writerow(info)

            with open('./MajorRevision/infov2.csv', 'a', newline='') as mycsvfile:
                wr = csv.writer(mycsvfile)
                wr.writerow(infov) 

            with open('./MajorRevision/infovPV2.csv', 'a', newline='') as mycsvfile:
                wr = csv.writer(mycsvfile)	
                wr.writerow(infoPV) 
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
            tmp[a1][0] = 1
            tmp[a2][1] = 1
            tmp[a3][2] = 1
            tmp[a4][3] = 1
            tmp[a5][4] = 1
            tmp[a6][5] = 1
            tmp[a7][6] = 1

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

            proba[a1][0] = prob_a1
            proba[a2][1] = prob_a2
            proba[a3][2] = prob_a3
            proba[a4][3] = prob_a4
            proba[a5][4] = prob_a5
            proba[a6][5] = prob_a6
            proba[a7][6] = prob_a7

            net1.put_data((state1, a1, rewards[0], state_next1, prob_a1, terminals[0]))
            net2.put_data((state2, a2, rewards[1], state_next2, prob_a2, terminals[1]))
            net3.put_data((state3, a3, rewards[2], state_next3, prob_a3, terminals[2]))
            net4.put_data((state4, a4, rewards[3], state_next4, prob_a4, terminals[3]))
            net5.put_data((state5, a5, rewards[4], state_next5, prob_a5, terminals[4]))
            net6.put_data((state6, a6, rewards[5], state_next6, prob_a6, terminals[5]))
            net7.put_data((state7, a7, rewards[6], state_next7, prob_a7, terminals[6]))
            # ===================================
            state = np.copy(next_states)
            if (t + 1) % 4 == 0 or t == EP_LEN - 1 and train_mode:
              net1.train_net()
              torch.save(net1.state_dict(),os.path.join(save_path,'net1_params.pkl'))
              net2.train_net()
              torch.save(net2.state_dict(),os.path.join(save_path,'net2_params.pkl'))
              net3.train_net()
              torch.save(net3.state_dict(),os.path.join(save_path,'net3_params.pkl'))
              net4.train_net()
              torch.save(net4.state_dict(),os.path.join(save_path,'net4_params.pkl'))
              net5.train_net()
              torch.save(net5.state_dict(),os.path.join(save_path,'net5_params.pkl'))
              net6.train_net()
              torch.save(net6.state_dict(),os.path.join(save_path,'net6_params.pkl'))
              net7.train_net()
              torch.save(net7.state_dict(),os.path.join(save_path,'net7_params.pkl'))

            #print("[actions]", actions)
            #print("[acts]", acts)
            # print("[temp ac]", tmp)
            # print(rewards)
        # Train network ================================================================================================
              


            # ==============================================================================================================
        if train_mode:
            with open('./MajorRevision/reward1.csv', 'a', newline='') as mycsvfile:
                wr = csv.writer(mycsvfile)
                wr.writerow([ep_r1,ep_r2,ep_r3,ep_r4,ep_r5,ep_r6,ep_r7])
        if EP_MAX%print_interval==0 and EP_MAX!=0:
            print("# of episode :{}, avg score : {}, epsilon : {}".format(ep, [ep_r1,ep_r2,ep_r3,ep_r4,ep_r5,ep_r6,ep_r7], epsilon))
else:
   normalizer_weights = np.load(save_path + '/normalizer_weight.npy')
   # print(normalizer_weights)
   normalizer.n = normalizer_weights[0]
   normalizer.mean = normalizer_weights[1]
   normalizer.mean_diff = normalizer_weights[2]
   normalizer.var = normalizer_weights[3]

   ra = 292
   rb = 4
   state = env.reset(ra+1,rb)
   ep_r1 = 0
   ep_r2 = 0
   ep_r3 = 0
   ep_r4 = 0
   ep_r5 = 0
   ep_r6 = 0
   ep_r7 = 0

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

      prob1 = net1.pi(torch.from_numpy(state1).float())
      prob2 = net2.pi(torch.from_numpy(state2).float())
      prob3 = net3.pi(torch.from_numpy(state3).float())
      prob4 = net4.pi(torch.from_numpy(state4).float())
      prob5 = net5.pi(torch.from_numpy(state5).float())
      prob6 = net6.pi(torch.from_numpy(state6).float())
      prob7 = net7.pi(torch.from_numpy(state7).float())

      m1 = Categorical(prob1)
      m2 = Categorical(prob2)
      m3 = Categorical(prob3)
      m4 = Categorical(prob4)
      m5 = Categorical(prob5)
      m6 = Categorical(prob6)
      m7 = Categorical(prob7)

      a1 = m1.sample().item()
      a2 = m2.sample().item()
      a3 = m3.sample().item()
      a4 = m4.sample().item()
      a5 = m5.sample().item()
      a6 = m6.sample().item()
      a7 = m7.sample().item()

      acts1 = a1 * 0.002 + 0.98     #tap1         1.00 ~ 1.04 
      acts2 = a2 * 0.1   - 1           #sop1(69-15) -0.12 ~ 0.24
      acts3 = a3 * 0.03  - 0.3         #shunt69 -0.6 ~ 0.6
      acts4 = a4 * 0.03  - 0.3         #shunt15 -1.1 ~ 0.1
      acts5 = a5 * 0.1   - 1           #sop2(27-54) -0.12 ~ 0.24
      acts6 = a6 * 0.03  - 0.3         #shunt69 -0.6 ~ 0.6
      acts7 = a7 * 0.03  - 0.3         #shunt69 -0.6 ~ 0.6
            
      print(acts1,acts2,acts3,acts4,acts5,acts6,acts7)


      next_states, rewards, terminals, ifo, infov, infoPV = env.step(acts1,acts2,acts3,acts4,acts5,acts6,acts7)

      with open(save_path +'/validact3.csv', 'a', newline='') as mycsvfile:
        wr = csv.writer(mycsvfile)
        wr.writerow(ifo)

      with open(save_path +'/infov.csv', 'a', newline='') as mycsvfile:
        wr = csv.writer(mycsvfile)
        wr.writerow(infov)
      
      with open(save_path +'/infoPV.csv', 'a', newline='') as mycsvfile:
        wr = csv.writer(mycsvfile)
        wr.writerow(infoPV)
      ep_r1 += rewards[0]
      ep_r2 += rewards[1]
      ep_r3 += rewards[2]
      ep_r4 += rewards[3]
      ep_r5 += rewards[4]
      ep_r6 += rewards[5]
      ep_r7 += rewards[6]

      state = np.copy(next_states)
   print(ep_r1, ep_r2, ep_r3,ep_r4, ep_r5, ep_r6, ep_r7)