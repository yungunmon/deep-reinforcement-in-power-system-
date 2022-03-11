import random
import numpy as np
import csv
from MultiCompanyAgent import IAC
from MultiCompanyEnv import env
from MultiCompanyNormalizer import Normalizer
import time
print("Major Revision Experiment Start")

save_path = "./MajorRevision"
train_mode = True
# Load model은 Agent_discre.py에서도 변경해줘야함
load_model = False

# Number of Agents
num_agents = 2
env = env(num_agents, load_model)
state_size = env.state_size
action_size = env.action_size

total_agents = num_agents
EP_MAX = 10000
EP_LEN = env.MaxTime
BATCH = env.batch_size

A_LR = 0.00002
C_LR = 0.0001

# Path of env
epsilon = 1.0
MaxEpsilon = 0.6
MinEpsilon = 0.001

net = IAC(save_path)

normalizer = Normalizer(state_size)

if train_mode:
    for ep in range(EP_MAX):

        epsilon -= 1/3000
        if epsilon < MinEpsilon:
            epsilon = MinEpsilon

        state = env.reset()
        ep_r1 = 0
        ep_r2 = 0

        buffer_s, buffer_s_next, buffer_a, buffer_r, buffer_t = [], [], [], [], []

        for t in range(EP_LEN):

            normalizer.observe(state[0])
            normalizer.observe(state[1])   

            state = normalizer.normalize(state)

            state = np.reshape(np.transpose(state), newshape=(-1, state_size, num_agents))
            actions = net.action_prob(state, False)


            # For Debug ================================================================
            # k = net.sess.run(net.k, feed_dict={net.state : state})
            # print("k", k)
            # print("state", state)

            actions = np.reshape(actions, newshape=(action_size, num_agents))
         
            if epsilon > np.random.rand():
                
                acts = np.random.randint(low=0, high=action_size, size=(num_agents,))  #(2,1)
                np.random.seed(int((time.time()-round(time.time())+1)*100))

                if acts[0]==0:
                    acts1 = 1.005
                elif acts[0]==1:
                    acts1 = 1.01
                elif acts[0]==2:
                    acts1 = 1.015
                elif acts[0]==3:
                    acts1 = 1.02
                elif acts[0]==4:
                    acts1 = 1.025
                elif acts[0]==5:
                    acts1 = 1.03
                elif acts[0]==6:
                    acts1 = 1.035
                elif acts[0]==7:
                    acts1 = 1.04

                if acts[1]==0:
                    acts2 = 0
                elif acts[1]==1:
                    acts2 = -0.1
                elif acts[1]==2:
                    acts2 = -0.2
                elif acts[1]==3:
                    acts2 = -0.3
                elif acts[1]==4:
                    acts2 = -0.4
                elif acts[1]==5:
                    acts2 = -0.5
                elif acts[1]==6:
                    acts2 = -0.6
                elif acts[1]==7:
                    acts2 = -0.7

            else:
                acts = np.argmax(actions, axis=0)
                 
                if acts[0]==0:
                    acts1 = 1.005
                elif acts[0]==1:
                    acts1 = 1.01
                elif acts[0]==2:
                    acts1 = 1.015
                elif acts[0]==3:
                    acts1 = 1.02
                elif acts[0]==4:
                    acts1 = 1.025
                elif acts[0]==5:
                    acts1 = 1.03
                elif acts[0]==6:
                    acts1 = 1.035
                elif acts[0]==7:
                    acts1 = 1.04

                if acts[1]==0:
                    acts2 = 0
                elif acts[1]==1:
                    acts2 = -0.1
                elif acts[1]==2:
                    acts2 = -0.2
                elif acts[1]==3:
                    acts2 = -0.3
                elif acts[1]==4:
                    acts2 = -0.4
                elif acts[1]==5:
                    acts2 = -0.5
                elif acts[1]==6:
                    acts2 = -0.6
                elif acts[1]==7:
                    acts2 = -0.7

            next_states, rewards, terminals, info,infos, infov = env.step(acts1,acts2)

            with open('./MajorRevision/info.csv', 'a', newline='') as mycsvfile:
                wr = csv.writer(mycsvfile)
                 
                wr.writerow(info)
            with open('./MajorRevision/infos.csv', 'a', newline='') as mycsvfile:
                wr = csv.writer(mycsvfile)
                 
                wr.writerow(infos)    
            with open('./MajorRevision/infov.csv', 'a', newline='') as mycsvfile:
                wr = csv.writer(mycsvfile)
                 
                wr.writerow(infov)        

            ep_r1 += rewards[0]
            ep_r2 += rewards[1]

            # Reconfiguration for action
            tmp = np.zeros(shape=(action_size, num_agents))

            if acts1 == 1.005 :
                acts1 = 0
            elif acts1 == 1.01:
                acts1 = 1
            elif acts1 == 1.015:
                acts1 = 2
            elif acts1 == 1.02:
                acts1 = 3
            elif acts1 == 1.025:
                acts1 = 4
            elif acts1 == 1.03:
                acts1 = 5
            elif acts1 == 1.035:
                acts1 = 6
            elif acts1 == 1.04:
                acts1 = 7

            if acts2 == 0 :
                acts2 = 0
            elif acts2 == -0.1:
                acts2 = 1
            elif acts2 == -0.2:
                acts2 = 2
            elif acts2 == -0.3:
                acts2 = 3
            elif acts2 == -0.4:
                acts2 = 4
            elif acts2 == -0.5:
                acts2 = 5
            elif acts2 == -0.6:
                acts2 = 6
            elif acts2 == -0.7:
                acts2 = 7         
            
            tmp[acts1][0] = 1
            tmp[acts2][1] = 1
            state_next = np.copy(next_states)
            state_next = normalizer.normalize(state_next)
            state_next = np.reshape(np.transpose(state_next), newshape=(-1, state_size, num_agents))

            #print("[next state]", next_states)
            #print("[state_next]", state_next)

            # Store transition ==================
            buffer_s.append(state)
            buffer_s_next.append(state_next)
            buffer_a.append(tmp)
            buffer_r.append(rewards)
            buffer_t.append(terminals)
            # ===================================

            state = np.copy(next_states)

            #print("[actions]", actions)
            #print("[acts]", acts)
            # print("[temp ac]", tmp)
            # print(rewards)

            # Train network ================================================================================================
            if (t + 1) % BATCH == 0 or t == EP_LEN - 1 and train_mode:
                # if train_mode and t == 1:
                bs = np.reshape(buffer_s, newshape=(-1, state_size, num_agents))
                bs_ = np.reshape(buffer_s_next, newshape=(-1, state_size, num_agents))
                ba = np.reshape(buffer_a, newshape=(-1, action_size, num_agents))
                br = np.reshape(buffer_r, newshape=(-1, num_agents))
                bt = np.reshape(buffer_t, newshape=(-1, num_agents))
             

                loss = net.train_op(s=bs, s_next=bs_, a=ba, r=br, t=bt, c_lr=C_LR, a_lr=A_LR)
                buffer_s, buffer_s_next, buffer_a, buffer_r, buffer_t = [], [], [], [], []
            # ==============================================================================================================

        if (ep + 1) % 10 == 0:
            net.save_model(save_path)
            # self.n = np.zeros(nb_inputs)
            # self.mean = np.zeros(nb_inputs)
            # self.mean_diff = np.zeros(nb_inputs)
            # self.var = np.zeros(nb_inputs)
            weights = [normalizer.n, normalizer.mean, normalizer.mean_diff, normalizer.var]
            np.save('./MajorRevision/normalizer_weight', weights)

        print("[Ep] {} [Reward] {} [A_Loss] {} [C_Loss] {} [Epsilon] {}".format(ep+1, ep_r1+ep_r2,  loss[0], loss[1], epsilon))
        if train_mode:
            with open('./MajorRevision/reward.csv', 'a', newline='') as mycsvfile:
                wr = csv.writer(mycsvfile)
                tmp = ep_r1 + ep_r2
                wr.writerow([ep_r1,ep_r2, tmp])
                
else:
    normalizer_weights = np.load(save_path + '/normalizer_weight.npy')
    # print(normalizer_weights)
    normalizer.n = normalizer_weights[0]
    normalizer.mean = normalizer_weights[1]
    normalizer.mean_diff = normalizer_weights[2]
    normalizer.var = normalizer_weights[3]

    state = env.reset()
    ep_r1 = 0
    ep_r2 = 0
    for t in range(EP_LEN):
        normalizer.observe(state[0])    
        normalizer.observe(state[1])    

        state = normalizer.normalize(state)
        state = np.reshape(np.transpose(state), newshape=(-1, state_size, num_agents))
        actions = net.action_prob(state, False)

        actions = np.reshape(actions, newshape=(action_size, num_agents))
        acts = np.argmax(actions, axis=0)

        if acts[0]==0:
            acts1 = -0.5
        elif acts[0]==1:
            acts1 = -0.4
        elif acts[0]==2:
            acts1 = -0.3
        elif acts[0]==3:
            acts1 = -0.2
        elif acts[0]==4:
            acts1 = 0.2
        elif acts[0]==5:
            acts1 = 0.3
        elif acts[0]==6:
            acts1 = 0.4
        elif acts[0]==7:
            acts1 = 0.5

        if 5*soc1 + acts1 > 4:
            acts1 = -0.5               

        elif 5*soc1 + acts1 < 1:
            acts1 = 0.5
        else :
            acts1 = acts1

        if acts[1]==0:
            acts2 = -0.5
        elif acts[1]==1:
            acts2 = -0.4
        elif acts[1]==2:
            acts2 = -0.3
        elif acts[1]==3:
            acts2 = -0.2
        elif acts[1]==4:
            acts2 = 0.2
        elif acts[1]==5:
            acts2 = 0.3
        elif acts[1]==6:
            acts2 = 0.4
        elif acts[1]==7:
            acts2 = 0.5

        if 2.5*soc2 + acts2 > 2:
            acts2 = -0.5
        elif 2.5*soc2 + acts2 < 0.5:
            acts2 = 0.5
        else :
            acts2 = acts2

       
        next_states, rewards, terminals, info, infov = env.step(acts1,acts2, soc1,soc2)

        ep_r1 += rewards[0]
        ep_r2 += rewards[1]

        # Reconfiguration for action
        tmp = np.zeros(shape=(action_size, num_agents))


        if acts1 == -0.5 :
            acts1 = 0
        elif acts1 == -0.4:
            acts1 = 1
        elif acts1 == -0.3:
            acts1 = 2
        elif acts1 == -0.2:
            acts1 = 3
        elif acts1 == 0.2:
            acts1 = 4
        elif acts1 == 0.3:
            acts1 = 5
        elif acts1 == 0.4:
            acts1 = 6
        elif acts1 == 0.5:
            acts1 = 7

        if acts2 == -0.5 :
            acts2 = 0
        elif acts2 == -0.4:
            acts2 = 1
        elif acts2 == -0.3:
            acts2 = 2
        elif acts2 == -0.2:
            acts2 = 3
        elif acts2 == 0.2:
            acts2 = 4
        elif acts2 == 0.3:
            acts2 = 5
        elif acts2 == 0.4:
            acts2 = 6
        elif acts2 == 0.5:
            acts2 = 7
        tmp[acts[0]][0] = 1
        tmp[acts[1]][1] = 1

        print("[Info] {} [Actions] {}".format(info, acts))
       
        state = np.copy(next_states)
        with open('./MajorRevision/check1.csv', 'a', newline='') as mycsvfile:
            wr = csv.writer(mycsvfile)             
            wr.writerow(info)
        with open('./MajorRevision/checkv1.csv', 'a', newline='') as mycsvfile:
            wr = csv.writer(mycsvfile)             
            wr.writerow(infov)

    print(ep_r1, ep_r2)
