import numpy as np
import torch
from torch.distributions import Categorical
from policy import DQNActor, CommNetActor, Critic
from copy import deepcopy
from torch.nn.functional import one_hot
import torch.nn as nn
class Agent:
    def __init__(self, args, id = None):
        self.id        = id
        self.args      = args
        self.n_actions = args.n_actions
        self.state_shape = args.state_dim
        self.obs_shape = args.obs_dim
        if id < args.n_Comm_agents : self.actor = CommNetActor(args)
        else                       : self.actor = DQNActor(args)
        self.actor_target = deepcopy(self.actor)
        print(self.actor_target)
        self.actor.cuda()
        self.actor_target.cuda()
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                     lr=self.args.lr) 
    def o_preprocessing(self, o, ava):
        
        if self.id < self.args.n_Comm_agents:
            agent_obs   = np.copy(o)
            IDset       = np.arange(self.args.n_agents)
            logic_IDset = IDset == self.id
            logic_IDset = np.where(logic_IDset == 0)[0]
            logic_IDset = np.hstack([[self.id],logic_IDset])
            agent_obs   = agent_obs[logic_IDset,:]
            agent_obs   = torch.FloatTensor(agent_obs)

        else : 
            agent_obs = torch.FloatTensor(o)[self.id]
        return agent_obs
    
    def _O_preprocessing(self, O):
        if self.id < self.args.n_Comm_agents:
            agent_obs   = np.copy(O)
            IDset       = np.arange(self.args.n_agents)
            logic_IDset = IDset == self.id
            logic_IDset = np.where(logic_IDset == 0)[0]
            logic_IDset = np.hstack([[self.id],logic_IDset])
            agent_obs   = agent_obs[:,logic_IDset,:]
            agent_obs   = torch.FloatTensor(agent_obs)
        else: 
            agent_obs = torch.FloatTensor(O[:,self.id,:])
        return agent_obs
    
    def get_target_action(self,O,AVA):
        agent_obs = self._O_preprocessing(O)
        action_dist = self.actor_target(agent_obs.cuda())
        actions   = self._choose_action_from_softmax(action_dist,AVA)
        return actions
    
    def _choose_action_from_softmax(self, action_dist, ava):
        ava = torch.FloatTensor(ava).cuda()
        prob   = torch.nn.functional.softmax(action_dist, dim=-1) * ava
        action = torch.argmax(prob,-1)
        return action
    
    def select_action(self, o, ava):
        agent_obs = self.o_preprocessing(o,ava)
        action_dist = self.actor(agent_obs.cuda())
        action  = self._choose_action_from_softmax(action_dist, ava)
        return action
    
    def train(self,O,A,RWD,O_PRIME):
        O        = self._O_preprocessing(O)
        O_PRIME  = self._O_preprocessing(O_PRIME)
        O        = torch.FloatTensor(O).cuda()
        O_PRIME  = torch.FloatTensor(O_PRIME).cuda()
        r        = torch.FloatTensor(RWD).squeeze().cuda().unsqueeze(-1)
        A        = torch.tensor(A.reshape(-1,1),dtype=torch.int64).cuda()
        Q        = self.actor(O).squeeze().gather(dim=-1, index=A)
        Q_TARGET = r + self.args.gamma * self.actor_target(O_PRIME).max(-1).values.detach().unsqueeze(-1)
        loss = nn.MSELoss()(Q, Q_TARGET.detach())
        self.actor_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.args.grad_norm_clip)
        self.actor_optimizer.step()
        return loss.item()
    

    def soft_update(self):
        for target_param, source_param in zip(self.actor_target.parameters(),
                                              self.actor.parameters()):
            target_param.data.copy_(
                (1 - self.args.tau) * target_param.data + self.args.tau * source_param.data)
