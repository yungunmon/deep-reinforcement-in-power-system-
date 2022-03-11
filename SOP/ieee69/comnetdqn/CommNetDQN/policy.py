import torch
import torch.nn as nn
import torch.nn.functional as f

class CommNetActor(nn.Module):
    def __init__(self, args):
        super(CommNetActor, self).__init__()
        self.encoding = nn.Linear(args.obs_dim, args.actor_dim) 
        self.cl1      = nn.Linear(args.actor_dim * 2, args.actor_dim)  
        self.cl2      = nn.Linear(args.actor_dim * 2, args.actor_dim)  
        self.cl3      = nn.Linear(args.actor_dim * 2, args.actor_dim)  
        self.cl4      = nn.Linear(args.actor_dim * 2, args.actor_dim) 
        self.decoding = nn.Linear(args.n_agents  * args.actor_dim , args.n_actions) 
        self.args = args
        self.input_shape = args.obs_dim
        
    def commlayer(self,commlayer, H):
        H      = H.reshape(-1, self.args.n_agents, self.args.actor_dim)
        C      = self.get_commvar(H)
        H_cat  = torch.cat([H,C],dim=-1)
        H_next = commlayer(H_cat)
        return H_next
    
    def get_commvar(self,H):
        C = H.reshape(-1, 1, self.args.n_agents * self.args.actor_dim)
        C = C.repeat(1, self.args.n_agents, 1)
        mask = (1 - torch.eye(self.args.n_agents)).cuda()
        mask = mask.view(-1, 1).repeat(1, self.args.actor_dim).view(self.args.n_agents, -1) 
        if self.args.cuda:
            mask = mask.cuda()
        C = C * mask.unsqueeze(0)
        C = C.reshape(-1, self.args.n_agents, self.args.n_agents, self.args.actor_dim)
        C = C.mean(dim=-2) 
        return C
    
    def forward(self, O): # Large O --> [o_own, o_oth1, o_oth2, ...]
        print('O',O)
        H0   = torch.sigmoid(self.encoding(O)).reshape(-1, self.args.n_agents, self.args.actor_dim)
        print('H0',H0)
        H1   = self.commlayer(self.cl1, H0)
        H2   = self.commlayer(self.cl2, H1)
        H3   = self.commlayer(self.cl3, H2)
        H4   = self.commlayer(self.cl4, H3)
        H    = H4.reshape(-1, self.args.n_agents * self.args.actor_dim)
        action_dist = self.decoding(H)
        #print('action_dist = ',action_dist)
        return action_dist

class DQNActor(nn.Module):
    # Because all the agents share the same network, input_shape=obs_shape+n_actions+n_agents
    def __init__(self, args):
        super(DQNActor, self).__init__()
        self.args = args
        self.encoding = nn.Linear(args.obs_dim, args.actor_dim) 
        self.fc1      = nn.Linear(args.actor_dim, args.actor_dim)  
        self.fc2      = nn.Linear(args.actor_dim, args.actor_dim)  
        self.fc3      = nn.Linear(args.actor_dim, args.actor_dim)  
        self.fc4      = nn.Linear(args.actor_dim, args.actor_dim) 
        self.decoding = nn.Linear(args.actor_dim, args.n_actions) 

    def forward(self, inputs):
        x = f.relu(self.encoding(inputs))
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = f.relu(self.fc3(x))
        x = f.relu(self.fc4(x))
        action_dist = f.relu(self.decoding(x))
        return action_dist


# Critic of Central-V
class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(args.state_dim+ args.n_agents*args.n_actions, args.critic_dim)
        self.fc2 = nn.Linear(args.critic_dim, args.critic_dim)
        self.fc3 = nn.Linear(args.critic_dim, args.critic_dim)
        self.fc4 = nn.Linear(args.critic_dim, 1)

    def forward(self, inputs):
        x = f.relu(self.fc1(inputs))
        x = f.relu(self.fc2(x))
        x = f.relu(self.fc3(x))
        q = self.fc4(x)
        return q
