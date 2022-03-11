from argument import get_args, get_env_info
from environmentComm import Environment
from agent import Agent
import torch
import copy
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
today = datetime.today().strftime("%Y-%m-%d-%H-%M-%S")

writer = SummaryWriter(f'evaluate/{today}')
args = get_args()
env = Environment()
args = get_env_info(args, env)
Agents = [Agent(args = args, id= i) for i in range(args.n_agents)]
t_sup     = np.zeros(args.episode_limit)
t_res     = np.zeros(args.episode_limit)
U_spt     = np.zeros((6,args.episode_limit))
for i in range(args.n_agents):
    checkpoint = torch.load(f'./model/actor{i}.tar')
    Agents[i].actor.load_state_dict(checkpoint['actor'])
    Agents[i].actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
for epoch in range(25):
    env.reset()
    o         = env.o # n_agents, obs_dim
    o_prime   = env.o_prime
    s         = env.s
    s_prime   = env.s_prime
    ava       = env.get_available_action()
    ava_prime = env.get_available_action()
    for t in range(args.episode_limit):
        o = o_prime.copy()
        actions, u_onehot = [], np.zeros((args.n_agents,args.n_actions))
        for i in range(args.n_agents):
            action   = Agents[i].select_action(o=o, ava= ava[i]).squeeze().data.cpu().numpy()
            actions.append(action)
        actions = np.array(actions)
        o, _, _, rewards, o_prime, _, ava, _ = env.step(actions)
        if t % 5 == 0:
            image = env.plot()
            writer.add_image(f'trajectory/EP{epoch+1}', image, t)
        writer.add_scalars(f'num_user/EP{epoch+1}', {
                    'total' : env.utility.SUPPORT[0],
                    'agent1': env.utility.SUPPORT[1],
                    'agent2': env.utility.SUPPORT[2],
                    'agent3': env.utility.SUPPORT[3],
                    'agent4': env.utility.SUPPORT[4],
                    'drone' : env.utility.SUPPORT[5]
                }, t)
        U_spt[:,t] += (env.utility.SUPPORT / 25)
        writer.add_scalars(f'support_rate/EP{epoch+1}',{'total':env.utility.T_support_rate}, t)
        writer.add_scalars(f'resolution/EP{epoch+1}'  ,{'total':env.utility.T_resolution}, t)
        t_sup[t] += (env.utility.T_support_rate / 25)
        t_res[t] += (env.utility.T_resolution /25)
        
for t in range(args.episode_limit):
    writer.add_scalars('Support_Rate',{'total':t_sup[t] }, t)
    writer.add_scalars('Resolution'  ,{'total':t_res[t] }, t)
    writer.add_scalars('Num_User', {
                    'total' : U_spt[0,t],
                    'agent1': U_spt[1,t],
                    'agent2': U_spt[2,t],
                    'agent3': U_spt[3,t],
                    'agent4': U_spt[4,t],
                    'drone' : U_spt[5,t]
                }, t)