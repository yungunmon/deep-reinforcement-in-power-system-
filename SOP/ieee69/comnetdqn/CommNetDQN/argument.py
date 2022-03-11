from easydict import EasyDict as edict

def get_args():
    args = edict({
        'seed':123,
        'batch_size' : 256,
        'actor_dim' : 64,
        'critic_dim' : 256,
        'train_epoch' : 100000,
        'cuda': True,
        'n_episodes':1,
        'gamma':0.99, 
        'tau':0.02,
        'optimizer':"RMS",
        'evaluate_cycle':5000,
        'evaluate_epoch':32,
        'lr' : 1e-3,
        'epsilon' : 1,
        'anneal_epsilon' : 0.001,
        'min_epsilon' : 0.01,
        'epsilon_anneal_scale' : 'episode',
        'td_lambda' : 0.8,
        'save_cycle' : 5000,
        'replay_capacity': 10000,
        'target_update_cycle' : 500,
        'grad_norm_clip' : 10,
        'k':2,
    })
    return args

def get_env_info(args,env):
    env_info           = env.get_info()
    args.episode_limit = env_info["episode_limit"]
    args.n_Comm_agents = env_info["n_Comm_agents"]
    args.n_DQN_agents  = env_info["n_DQN_agents"]
    args.n_agents      = env_info["n_agents"]
    args.n_actions     = env_info["n_actions"]
    args.state_dim     = env_info["state_dim"]
    args.obs_dim       = env_info["obs_dim"]
    return args