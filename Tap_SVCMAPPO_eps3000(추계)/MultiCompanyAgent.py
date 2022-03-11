import tensorflow as tf
import numpy as np
from MultiCompanyNetwork import CommNet
from MultiCompanyEnv import env

# Number of Agents
num_agents = 2
env = env(num_agents, False)
state_size = env.state_size
action_size = env.action_size
load_model = env.load_model

AGENT_NUM = num_agents

GAMMA = 0.9
LAMBDA = 0.95
EPSILON = 0.2


class IAC(object):
    def __init__(self, save_path):
        self.save_path = save_path
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.35
        self.sess = tf.Session(config=config)

        # Placeholder
        self.state = tf.placeholder(tf.float32, [None, state_size, AGENT_NUM], 'state')
        self.actions = tf.placeholder(tf.float32, [None, action_size, AGENT_NUM], 'actions')
        self.advantages = tf.placeholder(tf.float32, [None, 1, AGENT_NUM], 'advantages')

        # For Debug =====================================================================
        # self.k = tf.unstack(self.state, axis=-1)[0]

        # Hyperparamters for train
        self.c_lr = tf.placeholder(tf.float32)
        self.a_lr = tf.placeholder(tf.float32)
        self.is_train = tf.placeholder(tf.bool)

        with tf.variable_scope('critic'):
            # self.v :: [None, 1, agent_num]
            self.v, _ = CommNet(state=self.state, output_len=1, is_train=self.is_train)
            self.c_target = tf.placeholder(tf.float32, [None, 1, AGENT_NUM], 'critic_target')
            self.td_errors = self.c_target - self.v
            self.c_loss = tf.reduce_mean(tf.square(self.td_errors))
            c_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic')
            self.c_train = tf.train.AdamOptimizer(self.c_lr).minimize(self.c_loss, var_list=c_params)

        self.actor_out, actor_params = self.build_actor('actor', trainable=True)
        log_prob = tf.log(tf.maximum(self.actor_out * self.actions, 1e-6))
        # log_prob :: shape=(?, 4, 4) :: (?, action_size, num_agent)
        # advantages :: shape=(?, 1, 4) :: (?, value_size, num_agent)
        # surrogate :: shape=(?, 4, 4) :: (?, action_size, num_agent)
        surrogate = log_prob * self.advantages
        self.a_loss = - tf.reduce_mean(surrogate)
        self.a_train = tf.train.AdamOptimizer(self.a_lr).minimize(self.a_loss, var_list=actor_params)

        self.saver = tf.train.Saver()
        if load_model == True:
            ckpt = tf.train.get_checkpoint_state(save_path)
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("[Restore Model]")
        else:
            init = tf.global_variables_initializer()
            self.sess.run(init)
            print("[Initialize Model]")

    def build_actor(self, name, trainable):
        with tf.variable_scope(name):
            _, inter_out_list = CommNet(state=self.state, output_len=action_size, is_train=self.is_train)

            action_prob_list = []
            for temp_out in inter_out_list:
                action_prob_list.append(tf.nn.softmax(temp_out))
            action_prob = tf.stack(action_prob_list, axis=2)

        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return action_prob, params

    def action_prob(self, s, is_train):
        probs = self.sess.run(self.actor_out, {self.state: s,  self.is_train: is_train})
        return probs

    def get_v(self, s):
        return self.sess.run(self.v, {self.state: s, self.is_train: True})

    def train_op(self, s, s_next, a, r, t=None, c_lr=0.0001, a_lr=0.0001):

        batch_size = np.shape(s)[0]

        b_s = []
        b_a = []
        b_adv = []
        b_c_t = []
        b_s_ = []
        b_v = []
        b_delta = []

        # print("state_len ", state_len, "env_num", env_num)
        # print(np.shape(s))
        # print(np.shape(s_next))
        # print(np.shape(a))
        # print(np.shape(r))
        # print(np.shape(t))

        temp_s        = s
        temp_s_next   = s_next
        temp_a        = a
        temp_r        = r
        temp_t        = t

        b_s.append(temp_s)
        b_a.append(temp_a)

        v_next = self.sess.run(self.v, {self.state: temp_s_next,  self.is_train: False})


        c_t = np.zeros([batch_size, 1, AGENT_NUM])
        for idx in range(batch_size):
            if temp_t[idx].any():
                c_t[idx] = temp_r[idx]
                
            else:
                c_t[idx] = temp_r[idx] + GAMMA * v_next[idx]
                

        delta = self.sess.run(self.td_errors, {self.state: temp_s, self.c_target: c_t, self.is_train: False})

        adv = np.zeros([batch_size, 1, AGENT_NUM])
        adv[-1, 0] = delta[-1, 0]

        for idx in range(-2, -(batch_size + 1), -1):
            if temp_t[idx].any():
                adv[idx, 0] = delta[idx, 0]
            else:
                adv[idx, 0] = adv[idx + 1, 0] * GAMMA * LAMBDA + delta[idx, 0]

        b_c_t.append(c_t)
        b_adv.append(adv)

        b_s_.append(temp_s_next)
        b_v.append(v_next)
        b_delta.append(delta)

        # ==========================

        b_s = np.vstack(b_s)
        b_a = np.vstack(b_a)
        b_c_t = np.vstack(b_c_t)
        b_adv = np.vstack(b_adv)

        _, a_loss = self.sess.run([self.a_train, self.a_loss],
                                  {self.state: b_s, self.actions: b_a, self.advantages: b_adv, self.a_lr: a_lr,
                                   self.is_train: True})

        _, c_loss = self.sess.run([self.c_train, self.c_loss], {self.state: b_s, self.c_target: b_c_t, self.c_lr: c_lr,
                                                              self.is_train: True})
        return a_loss, c_loss

    def save_model(self, model_path, step=None):
        print("[Model save to]", model_path)
        save_path = self.saver.save(self.sess, model_path+ "/model.ckpt")
        return save_path

    def restore_model(self, model_path):
        print("[Model restored from]", model_path)
        self.saver.restore(self.sess, model_path)

if __name__ == "__main__":
    net = IAC()