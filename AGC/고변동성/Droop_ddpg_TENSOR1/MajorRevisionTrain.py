import random
import numpy as np
import csv
import collections
from MultiCompanyAgent import ddpg
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
import os


print("Major Revision Experiment Start")

save_path = "./MajorRevision"
train_mode = True
# Load model은 Agent_discre.py에서도 변경해줘야함
load_model = False

# Number of Agents
num_agents = 9
env = env(num_agents, load_model)
state_size = env.state_size
action_size = env.action_size

total_agents = num_agents
EP_MAX = 20000
EP_LEN = env.MaxTime
BATCH = env.batch_size
# Path of env
epsilon = 1.0
#MaxEpsilon = 0.6
MinEpsilon = 0.001
print_interval = 10

gamma        = 0.90
tau          = 0.005
def soft_update(net, net_target):
    for param_target, param in zip(net_target.parameters(), net.parameters()):
        param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)

class OrnsteinUhlenbeckNoise:
    def __init__(self, mu):
        self.theta, self.dt, self.sigma = 0.1, 0.01, 0.1
        self.mu = mu
        self.x_prev = np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=50000)

    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0 
            done_mask_lst.append([done_mask])
        
        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst, dtype=torch.float), \
                torch.tensor(r_lst, dtype=torch.float), torch.tensor(s_prime_lst, dtype=torch.float), \
                torch.tensor(done_mask_lst, dtype=torch.float)
    
    def size(self):
        return len(self.buffer)

def train(mu, mu_target, q, q_target, memory, q_optimizer, mu_optimizer):
    s,a,r,s_prime,done_mask  = memory.sample(15)
    
    target = r + gamma * q_target(s_prime, mu_target(s_prime)) * done_mask
    q_loss = F.smooth_l1_loss(q(s,a), target.detach())
    q_optimizer.zero_grad()
    q_loss.backward()
    q_optimizer.step()
    
    mu_loss = -q(s,mu(s)).mean() # That's all for the policy loss.
    mu_optimizer.zero_grad()
    mu_loss.backward()
    mu_optimizer.step()

device = torch.device("cuda")
net1 = ddpg(1)
net2 = ddpg(2)
net3 = ddpg(3)
net4 = ddpg(4)
net5 = ddpg(5)
net6 = ddpg(6)
net7 = ddpg(7)
net8 = ddpg(8)
net9 = ddpg(9)


net1buff = ReplayBuffer()
net2buff = ReplayBuffer()
net3buff = ReplayBuffer()
net4buff = ReplayBuffer()
net5buff = ReplayBuffer()
net6buff = ReplayBuffer()
net7buff = ReplayBuffer()
net8buff = ReplayBuffer()
net9buff = ReplayBuffer()
net1.q.load_state_dict(torch.load(os.path.join(save_path,'net1q_params.pkl')))
net2.q.load_state_dict(torch.load(os.path.join(save_path,'net2q_params.pkl')))
net3.q.load_state_dict(torch.load(os.path.join(save_path,'net3q_params.pkl')))
net4.q.load_state_dict(torch.load(os.path.join(save_path,'net4q_params.pkl')))
net5.q.load_state_dict(torch.load(os.path.join(save_path,'net5q_params.pkl')))
net6.q.load_state_dict(torch.load(os.path.join(save_path,'net6q_params.pkl')))
net7.q.load_state_dict(torch.load(os.path.join(save_path,'net7q_params.pkl')))
net8.q.load_state_dict(torch.load(os.path.join(save_path,'net8q_params.pkl')))
net9.q.load_state_dict(torch.load(os.path.join(save_path,'net9q_params.pkl')))

net1.mu.load_state_dict(torch.load(os.path.join(save_path,'net1mu_params.pkl')))
net2.mu.load_state_dict(torch.load(os.path.join(save_path,'net2mu_params.pkl')))
net3.mu.load_state_dict(torch.load(os.path.join(save_path,'net3mu_params.pkl')))
net4.mu.load_state_dict(torch.load(os.path.join(save_path,'net4mu_params.pkl')))
net5.mu.load_state_dict(torch.load(os.path.join(save_path,'net5mu_params.pkl')))
net6.mu.load_state_dict(torch.load(os.path.join(save_path,'net6mu_params.pkl')))
net7.mu.load_state_dict(torch.load(os.path.join(save_path,'net7mu_params.pkl')))
net8.mu.load_state_dict(torch.load(os.path.join(save_path,'net8mu_params.pkl')))
net9.mu.load_state_dict(torch.load(os.path.join(save_path,'net9mu_params.pkl')))


net1.qt.load_state_dict(torch.load(os.path.join(save_path,'net1qt_params.pkl')))
net2.qt.load_state_dict(torch.load(os.path.join(save_path,'net2qt_params.pkl')))
net3.qt.load_state_dict(torch.load(os.path.join(save_path,'net3qt_params.pkl')))
net4.qt.load_state_dict(torch.load(os.path.join(save_path,'net4qt_params.pkl')))
net5.qt.load_state_dict(torch.load(os.path.join(save_path,'net5qt_params.pkl')))
net6.qt.load_state_dict(torch.load(os.path.join(save_path,'net6qt_params.pkl')))
net7.qt.load_state_dict(torch.load(os.path.join(save_path,'net7qt_params.pkl')))
net8.qt.load_state_dict(torch.load(os.path.join(save_path,'net8qt_params.pkl')))
net9.qt.load_state_dict(torch.load(os.path.join(save_path,'net9qt_params.pkl')))

net1.mut.load_state_dict(torch.load(os.path.join(save_path,'net1mut_params.pkl')))
net2.mut.load_state_dict(torch.load(os.path.join(save_path,'net2mut_params.pkl')))
net3.mut.load_state_dict(torch.load(os.path.join(save_path,'net3mut_params.pkl')))
net4.mut.load_state_dict(torch.load(os.path.join(save_path,'net4mut_params.pkl')))
net5.mut.load_state_dict(torch.load(os.path.join(save_path,'net5mut_params.pkl')))
net6.mut.load_state_dict(torch.load(os.path.join(save_path,'net6mut_params.pkl')))
net7.mut.load_state_dict(torch.load(os.path.join(save_path,'net7mut_params.pkl')))
net8.mut.load_state_dict(torch.load(os.path.join(save_path,'net8mut_params.pkl')))
net9.mut.load_state_dict(torch.load(os.path.join(save_path,'net9mut_params.pkl')))
'''


net1.qt.load_state_dict(net1.q.state_dict())
net2.qt.load_state_dict(net2.q.state_dict())
net3.qt.load_state_dict(net3.q.state_dict())
net4.qt.load_state_dict(net4.q.state_dict())
net5.qt.load_state_dict(net5.q.state_dict())
net6.qt.load_state_dict(net6.q.state_dict())
net7.qt.load_state_dict(net7.q.state_dict())
net8.qt.load_state_dict(net8.q.state_dict())
net9.qt.load_state_dict(net9.q.state_dict())

net1.mut.load_state_dict(net1.mu.state_dict())
net2.mut.load_state_dict(net2.mu.state_dict())
net3.mut.load_state_dict(net3.mu.state_dict())
net4.mut.load_state_dict(net4.mu.state_dict())
net5.mut.load_state_dict(net5.mu.state_dict())
net6.mut.load_state_dict(net6.mu.state_dict())
net7.mut.load_state_dict(net7.mu.state_dict())
net8.mut.load_state_dict(net8.mu.state_dict())
net9.mut.load_state_dict(net9.mu.state_dict())

'''
lr_mu = 0.0003
lr_q = 0.0003
mu1_optimizer = optim.Adam(net1.mu.parameters(), lr=lr_mu)
mu2_optimizer = optim.Adam(net2.mu.parameters(), lr=lr_mu)
mu3_optimizer = optim.Adam(net3.mu.parameters(), lr=lr_mu)
mu4_optimizer = optim.Adam(net4.mu.parameters(), lr=lr_mu)
mu5_optimizer = optim.Adam(net5.mu.parameters(), lr=lr_mu)
mu6_optimizer = optim.Adam(net6.mu.parameters(), lr=lr_mu)
mu7_optimizer = optim.Adam(net7.mu.parameters(), lr=lr_mu)
mu8_optimizer = optim.Adam(net8.mu.parameters(), lr=lr_mu)
mu9_optimizer = optim.Adam(net9.mu.parameters(), lr=lr_mu)

q1_optimizer = optim.Adam(net1.q.parameters(), lr=lr_q)
q2_optimizer = optim.Adam(net2.q.parameters(), lr=lr_q)
q3_optimizer = optim.Adam(net3.q.parameters(), lr=lr_q)
q4_optimizer = optim.Adam(net4.q.parameters(), lr=lr_q)
q5_optimizer = optim.Adam(net5.q.parameters(), lr=lr_q)
q6_optimizer = optim.Adam(net6.q.parameters(), lr=lr_q)
q7_optimizer = optim.Adam(net7.q.parameters(), lr=lr_q)
q8_optimizer = optim.Adam(net8.q.parameters(), lr=lr_q)
q9_optimizer = optim.Adam(net9.q.parameters(), lr=lr_q)


ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(1))

normalizer = Normalizer(state_size)

if train_mode:
  for ep in range(EP_MAX):
    state , PV_, WT_= env.reset()
    ep_r1 = 0
    ep_r2 = 0
    ep_r3 = 0
    ep_r4 = 0
    ep_r5 = 0
    ep_r6 = 0
    ep_r7 = 0
    ep_r8 = 0
    ep_r9 = 0
        
    for t in range(EP_LEN):
      normalizer.observe(state[0])
      normalizer.observe(state[1])
      normalizer.observe(state[2])
      normalizer.observe(state[3])
      normalizer.observe(state[4])
      normalizer.observe(state[5])
      normalizer.observe(state[6])
      normalizer.observe(state[7])
      normalizer.observe(state[8])

      state = normalizer.normalize(state)
      state1 = state[0]
      state2 = state[1]
      state3 = state[2]
      state4 = state[3]
      state5 = state[4]
      state6 = state[5]
      state7 = state[6]
      state8 = state[7]
      state9 = state[8]

      a1 = net1.mu(torch.from_numpy(state1).float())
      a2 = net2.mu(torch.from_numpy(state2).float())
      a3 = net3.mu(torch.from_numpy(state3).float())
      a4 = net4.mu(torch.from_numpy(state4).float())
      a5 = net5.mu(torch.from_numpy(state5).float())
      a6 = net6.mu(torch.from_numpy(state6).float())
      a7 = net7.mu(torch.from_numpy(state7).float())
      a8 = net8.mu(torch.from_numpy(state8).float())
      a9 = net9.mu(torch.from_numpy(state9).float())
      print([a1.item(), a2.item(), a3.item(), a4.item(), a5.item(), a6.item(), a7.item(), a8.item(), a9.item()])

      a1 = np.clip(a1.item() + ou_noise()[0],-1,1)
      a2 = np.clip(a2.item() + ou_noise()[0],-1,1)
      a3 = np.clip(a3.item() + ou_noise()[0],-1,1)
      a4 = np.clip(a4.item() + ou_noise()[0],-1,1)
      a5 = np.clip(a5.item() + ou_noise()[0],-1,1)
      a6 = np.clip(a6.item() + ou_noise()[0],-1,1)
      a7 = np.clip(a7.item() + ou_noise()[0],-1,1)
      a8 = np.clip(a8.item() + ou_noise()[0],-1,1)
      a9 = np.clip(a9.item() + ou_noise()[0],-1,1)

      action = [a1, a2, a3, a4, a5, a6, a7, a8, a9] #ENV에서 
      next_states, rewards, terminals, PVcor, Wtcor, ifo , ifoGEN, ifoPVWT  = env.step(action,PV_, WT_)
      PV_ = PVcor
      WT_ = Wtcor

      '''

      with open('./MajorRevision/act3.csv', 'a', newline='') as mycsvfile:
          wr = csv.writer(mycsvfile)
          wr.writerow(ifo)
      with open('./MajorRevision/GEN3.csv', 'a', newline='') as mycsvfile:
          wr = csv.writer(mycsvfile)
          wr.writerow(ifoGEN)    
      with open('./MajorRevision/PVWT3.csv', 'a', newline='') as mycsvfile:
          wr = csv.writer(mycsvfile)
          wr.writerow(ifoPVWT)
      '''

      ep_r1 += rewards[0]
      ep_r2 += rewards[1]
      ep_r3 += rewards[2]
      ep_r4 += rewards[3]
      ep_r5 += rewards[4]
      ep_r6 += rewards[5]
      ep_r7 += rewards[6]
      ep_r8 += rewards[7]
      ep_r9 += rewards[8]

      # Reconfiguration for action
      state_next = np.copy(next_states)
      state_next = normalizer.normalize(state_next)
      state_next1 = state_next[0]
      state_next2 = state_next[1]
      state_next3 = state_next[2]
      state_next4 = state_next[3]
      state_next5 = state_next[4]
      state_next6 = state_next[5]
      state_next7 = state_next[6]
      state_next8 = state_next[7]
      state_next9 = state_next[8]
      
      state_next = np.copy(next_states)
      state_next = normalizer.normalize(state_next)
      state_next = np.reshape(np.transpose(state_next), newshape=(-1, state_size, num_agents))
      
      net1buff.put((state1, a1, rewards[0], state_next1, terminals[0]))
      net2buff.put((state2, a2, rewards[1], state_next2, terminals[1]))
      net3buff.put((state3, a3, rewards[2], state_next3, terminals[2]))
      net4buff.put((state4, a4, rewards[3], state_next4, terminals[3]))
      net5buff.put((state5, a5, rewards[4], state_next5, terminals[4]))
      net6buff.put((state6, a6, rewards[5], state_next6, terminals[5]))
      net7buff.put((state7, a7, rewards[6], state_next7, terminals[6]))
      net8buff.put((state8, a8, rewards[7], state_next8, terminals[7]))
      net9buff.put((state9, a9, rewards[8], state_next9, terminals[8]))

      # ===================================
      state = np.copy(next_states)

      #print("[actions]", actions)
      #print("[acts]", acts)
      # print("[temp ac]", tmp)
      # print(rewards)
      # Train network ================================================================================================
      if net9buff.size()>5000:
        for i in range(10):
          train(net1.mu, net1.mut, net1.q, net1.qt, net1buff, q1_optimizer, mu1_optimizer)
          train(net2.mu, net2.mut, net2.q, net2.qt, net2buff, q2_optimizer, mu2_optimizer)
          train(net3.mu, net3.mut, net3.q, net3.qt, net3buff, q3_optimizer, mu3_optimizer)
          train(net4.mu, net4.mut, net4.q, net4.qt, net4buff, q4_optimizer, mu4_optimizer)
          train(net5.mu, net5.mut, net5.q, net5.qt, net5buff, q5_optimizer, mu5_optimizer)
          train(net6.mu, net6.mut, net6.q, net6.qt, net6buff, q6_optimizer, mu6_optimizer)
          train(net7.mu, net7.mut, net7.q, net7.qt, net7buff, q7_optimizer, mu7_optimizer)
          train(net8.mu, net8.mut, net8.q, net8.qt, net8buff, q8_optimizer, mu8_optimizer)
          train(net9.mu, net9.mut, net9.q, net9.qt, net9buff, q9_optimizer, mu9_optimizer)
          soft_update(net1.mu, net1.mut)
          soft_update(net2.mu, net2.mut)
          soft_update(net3.mu, net3.mut)
          soft_update(net4.mu, net4.mut)
          soft_update(net5.mu, net5.mut)
          soft_update(net6.mu, net6.mut)
          soft_update(net7.mu, net7.mut)
          soft_update(net8.mu, net8.mut)
          soft_update(net9.mu, net9.mut)
          soft_update(net1.q , net1.qt)
          soft_update(net2.q , net2.qt)
          soft_update(net3.q , net3.qt)
          soft_update(net4.q , net4.qt)
          soft_update(net5.q , net5.qt)
          soft_update(net6.q , net6.qt)
          soft_update(net7.q , net7.qt)
          soft_update(net8.q , net8.qt)
          soft_update(net9.q , net9.qt)

      if ep % 100 == 0 : 
        torch.save(net1.mu.state_dict(),os.path.join(save_path,'net1mu_params.pkl'))
        torch.save(net2.mu.state_dict(),os.path.join(save_path,'net2mu_params.pkl'))
        torch.save(net3.mu.state_dict(),os.path.join(save_path,'net3mu_params.pkl'))
        torch.save(net4.mu.state_dict(),os.path.join(save_path,'net4mu_params.pkl'))
        torch.save(net5.mu.state_dict(),os.path.join(save_path,'net5mu_params.pkl'))
        torch.save(net6.mu.state_dict(),os.path.join(save_path,'net6mu_params.pkl'))
        torch.save(net7.mu.state_dict(),os.path.join(save_path,'net7mu_params.pkl'))
        torch.save(net8.mu.state_dict(),os.path.join(save_path,'net8mu_params.pkl'))
        torch.save(net9.mu.state_dict(),os.path.join(save_path,'net9mu_params.pkl'))
        torch.save(net1.q.state_dict(),os.path.join(save_path,'net1q_params.pkl'))
        torch.save(net2.q.state_dict(),os.path.join(save_path,'net2q_params.pkl'))
        torch.save(net3.q.state_dict(),os.path.join(save_path,'net3q_params.pkl'))
        torch.save(net4.q.state_dict(),os.path.join(save_path,'net4q_params.pkl'))
        torch.save(net5.q.state_dict(),os.path.join(save_path,'net5q_params.pkl'))
        torch.save(net6.q.state_dict(),os.path.join(save_path,'net6q_params.pkl'))
        torch.save(net7.q.state_dict(),os.path.join(save_path,'net7q_params.pkl'))
        torch.save(net8.q.state_dict(),os.path.join(save_path,'net8q_params.pkl'))
        torch.save(net9.q.state_dict(),os.path.join(save_path,'net9q_params.pkl'))
        torch.save(net1.mut.state_dict(),os.path.join(save_path,'net1mut_params.pkl'))
        torch.save(net2.mut.state_dict(),os.path.join(save_path,'net2mut_params.pkl'))
        torch.save(net3.mut.state_dict(),os.path.join(save_path,'net3mut_params.pkl'))
        torch.save(net4.mut.state_dict(),os.path.join(save_path,'net4mut_params.pkl'))
        torch.save(net5.mut.state_dict(),os.path.join(save_path,'net5mut_params.pkl'))
        torch.save(net6.mut.state_dict(),os.path.join(save_path,'net6mut_params.pkl'))
        torch.save(net7.mut.state_dict(),os.path.join(save_path,'net7mut_params.pkl'))
        torch.save(net8.mut.state_dict(),os.path.join(save_path,'net8mut_params.pkl'))
        torch.save(net9.mut.state_dict(),os.path.join(save_path,'net9mut_params.pkl'))
        torch.save(net1.qt.state_dict(),os.path.join(save_path,'net1qt_params.pkl'))
        torch.save(net2.qt.state_dict(),os.path.join(save_path,'net2qt_params.pkl'))
        torch.save(net3.qt.state_dict(),os.path.join(save_path,'net3qt_params.pkl'))
        torch.save(net4.qt.state_dict(),os.path.join(save_path,'net4qt_params.pkl'))
        torch.save(net5.qt.state_dict(),os.path.join(save_path,'net5qt_params.pkl'))
        torch.save(net6.qt.state_dict(),os.path.join(save_path,'net6qt_params.pkl'))
        torch.save(net7.qt.state_dict(),os.path.join(save_path,'net7qt_params.pkl'))
        torch.save(net8.qt.state_dict(),os.path.join(save_path,'net8qt_params.pkl'))
        torch.save(net9.qt.state_dict(),os.path.join(save_path,'net9qt_params.pkl'))



      # ==============================================================================================================
    if (ep + 1) % 10 == 0:
      # self.n = np.zeros(nb_inputs)
      # self.mean = np.zeros(nb_inputs)
      # self.mean_diff = np.zeros(nb_inputs)
      # self.var = np.zeros(nb_inputs)
      weights = [normalizer.n, normalizer.mean, normalizer.mean_diff, normalizer.var]
      np.save('./MajorRevision/normalizer_weight', weights)

    print("[Ep] {} [Reward] {}".format(ep+1, ep_r1))
    if train_mode:
      with open('./MajorRevision/reward2.csv', 'a', newline='') as mycsvfile:
          wr = csv.writer(mycsvfile)         
          wr.writerow([ep_r1])
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