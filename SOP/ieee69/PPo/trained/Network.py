import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import tensorflow as tf
import numpy as np
from MultiCompanyEnv import env

# Number of Agents
num_agents = 4
env = env(num_agents, False)
state_size = env.state_size
action_size = env.action_size
load_model = env.load_model

AGENT_NUM = num_agents
#Hyperparameters
learning_rate = 0.00003
gamma         = 0.7
lmbda         = 0.7
eps_clip      = 0.2
K_epoch       = 5
T_horizon     = 24

class PPO(nn.Module):
    def __init__(self, numb , path):
        super(PPO, self).__init__()
        self.data = []    
        self.id = str(numb)    
        self.fc1   = nn.Linear(state_size,256)
        self.fc2   = nn.Linear(256,256)
        self.fc_pi = nn.Linear(256,action_size)
        self.fc_v  = nn.Linear(256,1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.model_dir = path


    def pi(self, x, softmax_dim = 0):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc_pi(x)        
        prob = F.softmax(x, dim=softmax_dim)
        return prob
    
    def v(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        v = self.fc_v(x)
        return v
      
    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition
            
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
            
        s,a,r,s_prime,done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst, dtype=torch.int64), \
                                          torch.tensor(r_lst, dtype=torch.float), torch.tensor(s_prime_lst, dtype=torch.float), \
                                          torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst, dtype=torch.float)
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a
        
    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()

        for i in range(K_epoch):
            
            td_target = r + gamma * self.v(s_prime) * done_mask
            delta = td_target - self.v(s)
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = gamma * lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            pi = self.pi(s, softmax_dim=1)
            pi_a = pi.gather(1,a)
            
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))
            
            #print(pi_a,prob_a,ratio)
            
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s) , td_target.detach())
            #print(loss)
            #loss = loss.clone().detach().requires_grad_(True) 
            #print(loss1)
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        #torch.save(self.parameters(), self.model_dir + '/' + self.id + 'net_params.pkl')

