from argument import get_args, get_env_info
from environmentComm import Environment
from agent import Agent
from memory import ReplayBuffer, get_experience
import torch
import copy
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

args = get_args()
env = Environment()
args = get_env_info(args, env)
Agents = [Agent(args = args, id= i) for i in range(args.n_agents)]
replay = ReplayBuffer(args)
reward_record = []
today = datetime.today().strftime("%m%d%H%M%S")
info  = f'Comm{args.n_Comm_agents}_DNN{args.n_DQN_agents}_lr{args.lr}'
writer = SummaryWriter(f'runs/{today}+{info}')
for epoch in range(args.train_epoch):
    env.reset()
    o         = env.o # n_agents, obs_dim
    o_prime   = env.o_prime
    s         = env.s
    s_prime   = env.s_prime
    ava       = env.get_available_action()
    ava_prime = env.get_available_action()
    total_reward = 0.0
    TB_RWD = np.zeros(args.n_agents)
    TB_SR  = 0
    TB_SA  = 0
    TB_RES = 0
    TB_OL  = 0
    replay.buffer.clear()
    for t in range(args.episode_limit):
        o = o_prime.copy()
        actions, u_onehot = [], np.zeros((args.n_agents,args.n_actions))
        for i in range(args.n_agents):
            if np.random.rand()>args.epsilon:
                action   = Agents[i].select_action(o=o, ava= ava[i]).squeeze().data.cpu().numpy()
                print(Agents[i].select_action(o=o, ava= ava[i]))
            else:
                action   = np.random.randint(low=0,high=args.n_actions) 
            actions.append(action)
        actions = np.array(actions)
        o, _, _, rewards, o_prime, _, ava, _ = env.step(actions)
        experience = get_experience(o, actions, rewards, o_prime)
        replay.push(experience)
        TB_RWD +=rewards
        sr, sa, res, ol = env.utility.get_utils_info()
        TB_SR += sr
        TB_SA += sa
        TB_RES+= res
        TB_OL += ol
        # if (epoch+1) % 500 == 0:
        #     if t % 5 == 0:
        #         image = env.plot()
        #         writer.add_image(f'trajectory/EP{epoch+1}', image, t)
        #     writer.add_scalars(f'num_user/EP{epoch+1}', {
        #                 'total' : env.utility.SUPPORT[0],
        #                 'agent1': env.utility.SUPPORT[1],
        #                 'agent2': env.utility.SUPPORT[2],
        #                 'agent3': env.utility.SUPPORT[3],
        #                 'agent4': env.utility.SUPPORT[4],
        #                 'drone' : env.utility.SUPPORT[5]
        #             }, t)
        total_reward += rewards.sum()
    # [1] Sampling Batch Experiences
    args.epsilon -= args.anneal_epsilon
    args.epsilon  = max(args.epsilon, args.min_epsilon)
    samples = replay.sample()
    O = samples['O']
    ACTIONS = samples['A']
    REWARDS = samples['REWARDS']
    O_PRIME = samples['O_PRIME']
    ACTOR_LOSS = []
    for i in range(args.n_agents):
        loss = Agents[i].train(O,ACTIONS[:,i],REWARDS[:,i],O_PRIME)
        ACTOR_LOSS.append(loss)

    writer.add_scalars(f'loss', {
        'agent1': ACTOR_LOSS[0],
        'agent2': ACTOR_LOSS[1],
        'agent3': ACTOR_LOSS[2],
        'agent4': ACTOR_LOSS[3],
    }, epoch)

    writer.add_scalars(f'reward', {
            'total' : TB_RWD.sum(),
            'agent1': TB_RWD[0],
            'agent2': TB_RWD[1],
            'agent3': TB_RWD[2],
            'agent4': TB_RWD[3],
        }, epoch)
    writer.add_scalars(f'utility/support_rate', {
            'average' : TB_SR / args.episode_limit
        }, epoch)
    writer.add_scalars(f'utility/support_area', {
            'average' : TB_SA / args.episode_limit
        }, epoch)
    writer.add_scalars(f'utility/resolution', {
            'average' : TB_RES/ args.episode_limit
        }, epoch)
    writer.add_scalars(f'utility/overlapped', {
            'average' : TB_OL/ args.episode_limit
        }, epoch)

    for i in range(args.n_agents):
        torch.save({
                    'actor': Agents[i].actor.state_dict(),
                    'actor_optimizer': Agents[i].actor_optimizer.state_dict()
                    }, f'./actor{i}.tar')