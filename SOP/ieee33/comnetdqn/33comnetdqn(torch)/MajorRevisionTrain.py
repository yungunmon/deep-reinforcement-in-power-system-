import random
import numpy as np
import csv
from MultiCompanyEnv import env
from MultiCompanyNormalizer import Normalizer
from torch.autograd import Variable
from Network import CommNetActor
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import time
import os


print("Major Revision Experiment Start")

save_path = "./MajorRevision"
train_mode = True
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


net1 = CommNetActor(1,save_path)
net2 = CommNetActor(2,save_path)
net3 = CommNetActor(3,save_path)
net4 = CommNetActor(4,save_path)
'''
net1.load_state_dict(torch.load(os.path.join(save_path,'net1_params.pkl')))
net2.load_state_dict(torch.load(os.path.join(save_path,'net2_params.pkl')))
net3.load_state_dict(torch.load(os.path.join(save_path,'net3_params.pkl')))
net4.load_state_dict(torch.load(os.path.join(save_path,'net4_params.pkl')))
'''
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

            prob1 = net1.pi(torch.from_numpy(state).float())
            prob2 = net2.pi(torch.from_numpy(state).float())
            prob3 = net3.pi(torch.from_numpy(state).float())
            prob4 = net4.pi(torch.from_numpy(state).float())

            prob1 = prob1[0]
            prob2 = prob2[0]
            prob3 = prob3[0]
            prob4 = prob4[0]

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

            proba = np.zeros(shape=(action_size, num_agents))

            proba[a1][0] = prob_a1
            proba[a2][1] = prob_a2
            proba[a3][2] = prob_a3
            proba[a4][3] = prob_a4

            net1.put_data((state, a1, rewards[0], state_next, terminals[0]))
            net2.put_data((state, a2, rewards[1], state_next, terminals[1]))
            net3.put_data((state, a3, rewards[2], state_next, terminals[2]))
            net4.put_data((state, a4, rewards[3], state_next, terminals[3]))

            # ===================================
            state = np.copy(next_states)

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
        if (ep + 1) % 10 == 0:
            # self.n = np.zeros(nb_inputs)
            # self.mean = np.zeros(nb_inputs)
            # self.mean_diff = np.zeros(nb_inputs)
            # self.var = np.zeros(nb_inputs)
            weights = [normalizer.n, normalizer.mean, normalizer.mean_diff, normalizer.var]
            np.save('./MajorRevision/normalizer_weight', weights)            


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

   state = env.reset()
   ep_r1 = 0
   ep_r2 = 0 

   for t in range(EP_LEN):
      cost = state[0][5]
      state = normalizer.normalize(state)
      state = np.reshape(np.transpose(state), newshape=(-1, state_size, num_agents))
      actions = net.action_prob(state, False)

      actions = np.reshape(actions, newshape=(action_size, num_agents))
      acts = np.argmax(actions, axis=0)
      if acts[0]==0:
         acts1 = 30
      elif acts[0]==1:
         acts1 = 25
      elif acts[0]==2:
         acts1 = 20
      elif acts[0]==3:
         acts1 = 15
      elif acts[0]==4:
         acts1 = 10
      elif acts[0]==5:
         acts1 = 5
      elif acts[0]==6:
         acts1 = 2
      elif acts[0]==7:
         acts1 = 1         

            

      if acts[1]==0:
         acts2 = 20
      elif acts[1]==1:
         acts2 = 15
      elif acts[1]==2:
         acts2 = 10
      elif acts[1]==3:
         acts2 = 5
      elif acts[1]==4:
         acts2 = 4
      elif acts[1]==5:
         acts2 = 3
      elif acts[1]==6:
         acts2 = 2
      elif acts[1]==7:
         acts2 = 1


      next_states, rewards, terminals, info, infov, infoPV = env.step(acts1,acts2,cost)

      ep_r1 += rewards[0]
      ep_r2 += rewards[1]
      ep_r3 += rewards[2]

      # Reconfiguration for action
      tmp = np.zeros(shape=(action_size, num_agents))
      if acts1 == 30:
         acts1 = 0
      elif acts1 == 25:
         acts1 = 1
      elif acts1 == 20:
         acts1 = 2
      elif acts1 == 15:
         acts1 = 3
      elif acts1 == 10:
         acts1 = 4
      elif acts1 == 5:
         acts1 = 5
      elif acts1 == 2:
         acts1 = 6
      elif acts1 == 1:
         acts1 = 7


      if acts2 == 20 :
         acts2 = 0
      elif acts2 == 15:
         acts2 = 1
      elif acts2 == 10:
         acts2 = 2
      elif acts2 == 5:
         acts2 = 3
      elif acts2 == 4:
         acts2 = 4
      elif acts2 == 3:
         acts2 = 5
      elif acts2 == 2:
         acts2 = 6
      elif acts2 == 1:
         acts2 = 7      
    
         
      tmp[acts1][0] = 1
      tmp[acts2][1] = 1
      

      print("[Info] {} [Actions] {}".format(info, acts))
      a = info[0]
      b = info[1]
      c = info[2]
      d = info[3]
      e = info[4]
      f = info[5]
      g = info[6]
      state = np.copy(next_states)
      with open('./MajorRevision/validation.csv', 'a', newline='') as mycsvfile:
          wr = csv.writer(mycsvfile)
          wr.writerow([a,b,c,d,e,f,g])
   print(ep_r1, ep_r2, ep_r3 )