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
obs_dim =  state_size
AGENT_NUM = num_agents
#Hyperparameters
actor_dim     = 128
learning_rate = 0.00003
gamma         = 0.9
lmbda         = 0.9
eps_clip      = 0.2
K_epoch       = 5
T_horizon     = 24

class CommNetActor(nn.Module):
    def __init__(self, numb , path):
        super(CommNetActor, self).__init__()        
        self.data = []    
        self.id = str(numb)    
        self.encoding = nn.Linear(obs_dim, actor_dim) 
        self.cl1      = nn.Linear(actor_dim * 2, actor_dim)  
        self.cl2      = nn.Linear(actor_dim * 2, actor_dim)  
        self.cl3      = nn.Linear(actor_dim * 2, actor_dim)  
        self.cl4      = nn.Linear(actor_dim * 2, actor_dim) 

        self.fc1      = nn.Linear(actor_dim, actor_dim)  
        self.fc2      = nn.Linear(actor_dim, actor_dim)  
        self.fc3      = nn.Linear(actor_dim, actor_dim) 
        self.fc4      = nn.Linear(actor_dim, 1)
        self.decoding = nn.Linear(num_agents* actor_dim , action_size) 
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
    def commlayer(self,commlayer, H):
        H      = H.reshape(-1, num_agents, actor_dim)
        C      = self.get_commvar(H)
        H_cat  = torch.cat([H,C],dim=-1)
        H_next = commlayer(H_cat)
        return H_next

    def v(self, inputs):
        x = F.relu(self.encoding(inputs))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q = self.fc4(x)
        q = q.mean(dim=-1) 
        return q
    
    def get_commvar(self,H):
        C = H.reshape(-1, 1, num_agents * actor_dim)
        C = C.repeat(1, num_agents, 1)
        mask = (1 - torch.eye(num_agents))
        mask = mask.view(-1, 1).repeat(1, actor_dim).view(num_agents, -1) 
        C = C * mask.unsqueeze(0)
        C = C.reshape(-1, num_agents, num_agents, actor_dim)
        C = C.mean(dim=-2) 
        return C
    
    def pi(self, O): # Large O --> [o_own, o_oth1, o_oth2, ...]
        O = torch.FloatTensor(O)

        H0   =  torch.sigmoid(self.encoding(O)).reshape(-1, num_agents, actor_dim)
        H1   = F.relu(self.fc1( H0))
        H2   = F.relu(self.fc2( H1))
        H3   = F.relu(self.fc3( H2))
        H4   = self.commlayer(self.cl4, H3)
        H    = H4.reshape(-1, num_agents * actor_dim)
        action_dist = self.decoding(H)
        action_dist = torch.nn.functional.softmax(action_dist, dim=-1)
        #print(action_dist)
        return action_dist

    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []
        for transition in self.data:
            s,a,r,s_prime,done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r/100.0])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0
            done_lst.append([done_mask])
        
        s_batch, a_batch, r_batch, s_prime_batch, done_batch = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                                               torch.tensor(r_lst, dtype=torch.float), torch.tensor(s_prime_lst, dtype=torch.float), \
                                                               torch.tensor(done_lst, dtype=torch.float)
        self.data = []
        return s_batch, a_batch, r_batch, s_prime_batch, done_batch
        
    def train_net(self):
        s, a, r, s_prime, done = self.make_batch()
        td_target = r + gamma * self.v(s_prime) * done
        delta = td_target - self.v(s)
        
        pi = self.pi(s)
        pi_a = pi.gather(1,a)
        loss = -torch.log(pi_a) * delta.detach() + F.smooth_l1_loss(self.v(s), td_target.detach())

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()   

        #torch.save(self.parameters(), self.model_dir + '/' + self.id + 'net_params.pkl')

