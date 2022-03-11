import tensorflow as tf

import numpy as np 
from keras.models import Model 
from keras.layers import Dense, Input , Lambda 

class Actor(object) : 
    ##A2C 의 액터 신경망
    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate): 
        self.sess = sess 
        self.state_dim = state_d1m 
        self.action_dim = action_dim 
        self.action_bound = action_bound 
        self.learning_rate = learning_rate 
        
        # 표준편차의 최솟값과 최댓값 설정 
        self.std_bound = [ 1e-2 , 1.0 ] 
        
        # 액터 신경망 생성 
        self.model , self.theta , self.states = self.build_network() 
        
        # 실함수와 그래디언트 
        self.actions = tf.placeholder(tf.float32 , [ None , self.action_dim]) 
        self.advantages = tf.placeholder(tf.float32, [None,1]) 
        mu_a, std_a = self.model.output 
        log_policy_pdf = self.log_pdf(mu_a, std_a, self.actions) 
        loss_policy = log_policy_pdf * self.advantages 
        loss = tf.reduce_sum( -loss_policy) 
        dj_dtheta = tf.gradients(loss, self.theta) 
        grads = zip( dj_dtheta, self.theta) 
        self.actor_optimizer = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads) 

    ## 액터 신경망 
    def build_network(self): 
        state_input = Input((self. state_dim,)) 
        h1 = Dense(64, activation='relu')(state_input ) 
        h2 = Dense(32, activation='relu')(h1)
        h3 = Dense(16, activation='relu')(h2)
        out_mu = Dense(self.action_dim, activation='tanh' )(h3) 
        std_output = Dense(self.action_dim, activation='softplus')(h3) 

        ## 평균값을 [-action_bound, action_bound ] 법위로 조정 
        mu_output = Lambda(lambda x : x*self . action_bound )(out_mu )
        model = Model( state_input , [ mu_output , std_output ] ) 
        model.summary()
        return model , model.trainable_weights , state_input 
    ## 로그-정책 확률밀도함수 
    def log_pdf(self, mu, std, action):
        std = tf.clip_by_value( std , self. std_bound[0], self.std_bound[1])
        var = std**2
        log_policy_pdf = -0.5 * (action - mu ) ** 2 / var - 0.5 * tf.log(var * 2 * np.pi)
        return tf.reduce_sum(log_policy_pdf, 1, keepdims=True) 

    ## 액터 신경망 출력에서 확률적으로 행동을 추출 
    def get_action(self, state):
        mu_a, std_a = self.model.predict(np.reshape( state, [1,self.state_dim]))
        mu_a = mu_a[0]
        std_a = std_a[0]
        std_a = np.clip(std_a, self.std_bound[0], self. std_bound[1])
        action = np.random.normal(mu_a, std_a, size=self.action_dim)
        return action

    ## 액터 신경망에서 평균값 계산 
    def predict(self, state): 
        mu_a , _= self.model.predict(np.reshape(state,[1,self.state_dim])) #확인필요
        return mu_a[0] 
    ## 액터 신경망 학습 
    def train(self, states, actions, advantages) : 
        self.sess.run(self.actor_optimizer,feed_dict={
            self.states: states,
            self.actions : actions,
            self.advantages: advantages})

    ## 액터 신경망 파라미터 저장 
    def save_weights(self,path): 
        self.model.save_weights(path)

    ## 액티 신경망 파라미터 드 
    def load_weights(self,path):
        self.model.load_weights(path + 'pendulum_actor.h5')