import numpy as np
import threading

def get_experience(o,actions, rewards, o_prime):
    experience = {
            'o': o,
            'a' : actions,
            'rewards': rewards, 
            'o_prime': o_prime,
        }
    return experience
class ReplayBuffer:
    def __init__(self, args):
        self.args = args
        self.buffer = []
        
        # store the episode
    def push(self, experience):
        self.buffer.append(experience)
        if len(self.buffer) > self.args.replay_capacity:
            del self.buffer[0]
            
    def sample(self):
        idx = np.arange(len(self.buffer))
        np.random.shuffle(idx)
        idx=idx[:self.args.batch_size]
        O,ACTIONS, REWARDS,O_PRIME = [],[],[],[]
        for i in idx:
            O.append(self.buffer[i]['o'])
            ACTIONS.append(self.buffer[i]['a'])
            REWARDS.append(self.buffer[i]['rewards'])
            O_PRIME.append(self.buffer[i]['o_prime'])
        O = np.array(O)
        ACTIONS = np.array(ACTIONS)
        REWARDS = np.array(REWARDS)
        O_PRIME = np.array(O_PRIME)
        samples = dict({
            'O' : O,
            'A' : ACTIONS,
            'REWARDS' : REWARDS,
            'O_PRIME' : O_PRIME,
        })
        
        return samples
    