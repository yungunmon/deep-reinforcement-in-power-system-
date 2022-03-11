import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import tensorflow as tf
import numpy as np
from MultiCompanyEnv import env

# Number of Agents
num_agents = 9
env = env(num_agents, False)
state_size = env.state_size
action_size = env.action_size
load_model = env.load_model

AGENT_NUM = num_agents
#Hyperparameters
learning_rate = 0.000003
gamma         = 0.95
lmbda         = 0.9
eps_clip      = 0.1
K_epoch       = 5
T_horizon     = 24

class MuNet(nn.Module):
    def __init__(self):
        super(MuNet, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,64)
        self.fc_mu = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        mu = torch.tanh(self.fc_mu(x)) #action space is [-1,1]
        return mu        

class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        self.fc_s = nn.Linear(state_size, 128)
        self.fc_a = nn.Linear(1,128)
        self.fc_q = nn.Linear(256, 64)
        self.fc_out = nn.Linear(64,1)

    def forward(self, x, a):
        h1 = torch.tanh(self.fc_s(x))
        h2 = torch.tanh(self.fc_a(a))
        cat = torch.cat([h1,h2], dim=1)
        q = torch.tanh(self.fc_q(cat))
        q = self.fc_out(q)
        return q

class ddpg():  
    def __init__(self,num):        
        self.mu  = MuNet()
        self.mut = MuNet()
        self.q   = QNet()
        self.qt  = QNet()      
        self.agentnum = num  

    

        #torch.save(self.parameters(), self.model_dir + '/' + self.id + 'net_params.pkl')

