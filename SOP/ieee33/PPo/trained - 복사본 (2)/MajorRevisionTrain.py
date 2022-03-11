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
num_agents = 4
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

net1.load_state_dict(torch.load(os.path.join(save_path,'net1_params.pkl')))
net2.load_state_dict(torch.load(os.path.join(save_path,'net2_params.pkl')))
net3.load_state_dict(torch.load(os.path.join(save_path,'net3_params.pkl')))
net4.load_state_dict(torch.load(os.path.join(save_path,'net4_params.pkl')))
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

        for t in range(EP_LEN):

            normalizer.observe(state[0])
            normalizer.observe(state[1])
            normalizer.observe(state[2])
            normalizer.observe(state[3])

            state = normalizer.normalize(state)
            state1 = state[0]
            state2 = state[1]
            state3 = state[2]
            state4 = state[3]

            prob1 = net1.pi(torch.from_numpy(state1).float())
            prob2 = net2.pi(torch.from_numpy(state2).float())
            prob3 = net3.pi(torch.from_numpy(state3).float())
            prob4 = net4.pi(torch.from_numpy(state4).float())

            m1 = Categorical(prob1)
            m2 = Categorical(prob2)
            m3 = Categorical(prob3)
            m4 = Categorical(prob4)

            a1 = m1.sample().item()
            a2 = m2.sample().item()
            a3 = m3.sample().item()
            a4 = m4.sample().item()

            prob_a1 = prob1[a1].item()
            prob_a2 = prob2[a2].item()
            prob_a3 = prob3[a3].item()
            prob_a4 = prob4[a4].item()

            acts1 = a1 * 0.002 + 0.98     #tap1         1.00 ~ 1.04 
            acts2 = a2 * 0.1   - 1           #sop1(69-15) -1 ~ 1
            acts3 = a3 * 0.1  - 1         #shunt69 -0.3 ~ 0.3
            acts4 = a4 * 0.1  - 1         #shunt15 -0.3 ~ 0.3

            print(acts1,acts2,acts3,acts4)

            next_states, rewards, terminals, info, infov, infoPV = env.step(acts1,acts2, acts3,acts4)
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

            # Reconfiguration for action
            tmp = np.zeros(shape=(action_size, num_agents))
            tmp[a1][0] = 1
            tmp[a2][1] = 1
            tmp[a3][2] = 1
            tmp[a4][3] = 1

            state_next = np.copy(next_states)
            state_next = normalizer.normalize(state_next)
            state_next1 = state_next[0]
            state_next2 = state_next[1]
            state_next3 = state_next[2]
            state_next4 = state_next[3]

            state_next = np.reshape(np.transpose(state_next), newshape=(-1, state_size, num_agents))
            proba = np.zeros(shape=(action_size, num_agents))

            proba[a1][0] = prob_a1
            proba[a2][1] = prob_a2
            proba[a3][2] = prob_a3
            proba[a4][3] = prob_a4

            net1.put_data((state1, a1, rewards[0], state_next1, prob_a1, terminals[0]))
            net2.put_data((state2, a2, rewards[1], state_next2, prob_a2, terminals[1]))
            net3.put_data((state3, a3, rewards[2], state_next3, prob_a3, terminals[2]))
            net4.put_data((state4, a4, rewards[3], state_next4, prob_a4, terminals[3]))

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

            #print("[actions]", actions)
            #print("[acts]", acts)
            # print("[temp ac]", tmp)
            # print(rewards)
        # Train network ================================================================================================
        if (ep + 1) % 10 == 1:
            # self.n = np.zeros(nb_inputs)
            # self.mean = np.zeros(nb_inputs)
            # self.mean_diff = np.zeros(nb_inputs)
            # self.var = np.zeros(nb_inputs)
            weights = [normalizer.n, normalizer.mean, normalizer.mean_diff, normalizer.var]
            np.save('./MajorRevision/normalizer_weight', weights)
            # ==============================================================================================================
              


            # ==============================================================================================================
        if train_mode:
            with open('./MajorRevision/reward1.csv', 'a', newline='') as mycsvfile:
                wr = csv.writer(mycsvfile)
                wr.writerow([ep_r1,ep_r2,ep_r3,ep_r4])
        if EP_MAX%print_interval==0 and EP_MAX!=0:
            print("# of episode :{}, avg score : {}, epsilon : {}".format(ep, [ep_r1,ep_r2,ep_r3,ep_r4], epsilon))
else:
   normalizer_weights = np.load(save_path + '/normalizer_weight.npy')
   # print(normalizer_weights)
   normalizer.n = normalizer_weights[0]
   normalizer.mean = normalizer_weights[1]
   normalizer.mean_diff = normalizer_weights[2]
   normalizer.var = normalizer_weights[3]

   ra = 10
   rb = 4
   state = env.reset(ra+1,rb)
   ep_r1 = 0
   ep_r2 = 0
   ep_r3 = 0
   ep_r4 = 0

   for t in range(EP_LEN):

      normalizer.observe(state[0])
      normalizer.observe(state[1])
      normalizer.observe(state[2])
      normalizer.observe(state[3])

      state = normalizer.normalize(state)
      state1 = state[0]
      state2 = state[1]
      state3 = state[2]
      state4 = state[3]

      prob1 = net1.pi(torch.from_numpy(state1).float())
      prob2 = net2.pi(torch.from_numpy(state2).float())
      prob3 = net3.pi(torch.from_numpy(state3).float())
      prob4 = net4.pi(torch.from_numpy(state4).float())
      m1 = Categorical(prob1)
      m2 = Categorical(prob2)
      m3 = Categorical(prob3)
      m4 = Categorical(prob4)

      a1 = m1.sample().item()
      a2 = m2.sample().item()
      a3 = m3.sample().item()
      a4 = m4.sample().item()

      acts1 = a1 * 0.002 + 0.98     #tap1         1.00 ~ 1.04 
      acts2 = a2 * 0.1   - 1           #sop1(69-15) -0.12 ~ 0.24
      acts3 = a3 * 0.1   - 1           #shunt69 -0.6 ~ 0.6
      acts4 = a4 * 0.1   - 1           #shunt15 -1.1 ~ 0.1
            
      print(acts1,acts2,acts3,acts4)


      next_states, rewards, terminals, ifo, infov, infoPV = env.step(acts1,acts2,acts3,acts4)

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

      state = np.copy(next_states)
   print(ep_r1, ep_r2, ep_r3,ep_r4)