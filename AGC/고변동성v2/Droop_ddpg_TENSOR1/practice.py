import csv
import pandas as pd
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import tensorflow as tf
import numpy as np
from MultiCompanyEnv import env
'''
LOAD = pd.read_csv('./LOAD.csv')
#LOAD['date'] = pd.to_datetime(LOAD['date'])
#LOAD_= pd.DataFrame()
#LOAD_['date'] = pd.date_range(start='2021-03-14 00:00:00', end='2021-03-14 23:59:00', freq='MIN')
#LOAD_ = pd.merge(LOAD_, LOAD, on='date', how='outer')
#LOAD_ = LOAD_.interpolate()
#LOAD_.to_csv('LOAD.csv', index=False)
LOAD = LOAD['load']
LOAD = np.array(LOAD[0:1440])
LOAD = np.transpose(LOAD)
LOAD = LOAD/52000


def normal_pdf(x, mu=0, sigma=1):
    sqrt_two_pi = math.sqrt(2 * math.pi)
    return (math.exp(-(x-mu)**2 / 2 / sigma**2) / (sqrt_two_pi * sigma))

xs1 = [x/144 for x in range(-12*60, 12*60)]
pv1 = np.array([normal_pdf(x,sigma=1) for x in xs1]) 

PV = pd.read_csv('./PV.csv')
PV = np.array(PV[0:1440])
PV = np.transpose(PV)
PV = PV[0]/50000000

# 1440 일때

WT = pd.read_csv('./WT.csv')
        WT = np.array(WT[0:1440])
        WT = np.transpose(WT)
        self.Wind = WT[0]/8

        LOAD = pd.read_csv('./LOAD.csv')
        LOAD = LOAD['load']
        LOAD = np.array(LOAD[0:1440])
        LOAD = np.transpose(LOAD)
        self.LOAD = LOAD/52000

        PV = pd.read_csv('./PV.csv')
        PV = np.array(PV[0:1440])
        PV = np.transpose(PV)
        PV = PV[0]/50000000
        self.PV = PV
        '''
env = env(9, False)
state_size = env.state_size
  
class MuNet(nn.Module):
    def __init__(self):
        super(MuNet, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_mu = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.fc_mu(x))*2 # Multipled by 2 because the action space of the Pendulum-v0 is [-2,2]
        return mu

class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        self.fc_s = nn.Linear(state_size, 64)
        self.fc_a = nn.Linear(1,64)
        self.fc_q = nn.Linear(128, 32)
        self.fc_out = nn.Linear(32,1)

    def forward(self, x, a):
        h1 = F.relu(self.fc_s(x))
        h2 = F.relu(self.fc_a(a))
        cat = torch.cat([h1,h2], dim=1)
        q = F.relu(self.fc_q(cat))
        q = self.fc_out(q)
        return q

class ddpg():  
    def __init__(self):
        self.mu = MuNet()
        self.mut =MuNet()
        self.q = QNet()
        self.qt = QNet()

