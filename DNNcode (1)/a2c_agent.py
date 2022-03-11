import numpy as np 
import tensorflow as tf

import keras.backend as K 
import matplotlib.pyplot as plt 
from a2c_actor import Actor 
from a2c_critic import Critic 


class A2Cagent(object):
    def __init__(self,env):
        # 텐서플로 세션 설정 
        self.sess = tf.Session()
        K.set_session(self.sess)
        # 하이퍼파라미터 
        self.GAMMA = 0.95
        self.BATCH_SIZE = 32
        self.ACTOR_LEARNING_RATE = 0.0001
        self.CRITIC_LEARNING_RATE = 0.001
        
        # 환경
        self.env = env
        
        # 상태변수 차원 (dimension)
        self.state_dim = env.observation_space.shape[0]
        
        # 행동 차원 (dimension )
        self.action_dim = env.action_space.shape[0]
        # 행동의 최대 크기 
        self.action_bound = env.action_space.high[0]
        
        # 엑터 신경망 및 크리틱 신경망 생성
        self.actor = Actor(self.sess, self.state_dim, self.action_dim,
                           self.action_bound, self.ACTOR_LEARNING_RATE)
        self.critic = Critic(self.state_dim, self.action_dim,
                             self.CRITIC_LEARNING_RATE)
        # 그래디언트 계산을 워한 초가화 
        self.sess.run (tf.global_variables_initializer())
        
        # 에피소드에서 얻은 총 보상값을 저장하기 위한 변수 
        self.save_epi_reward = [] 
    ## 어드밴티지와 TD 타깃 계산 
    def advantage_td_target(self,reward,v_value, next_v_value, done):
        if done:
            y_k = reward
            advantage = y_k - v_value
        else:
            y_k = reward + self.GAMMA * next_v_value
            advantage = y_k - v_value
        return advantage, y_k
    ## 배치에 저장된 데이터 추출 
    def unpack_batch(self, batch):
        unpack = batch[0]
        for idx in range(len(batch)-1 ):
            unpack = np.append(unpack, batch[idx+1], axis=0)
        return unpack
    ## 에이전트 학습 
    def train(self, max_episode_num):
        # 에피소드미다 다음을 반복
        for ep in range(int(max_episode_num)):
            # 배치 초기화
            batch_state, batch_action, batch_td_target, batch_advantage = [], [], [], []
            # 에피소드 초기화
            time, episode_reward, done = 0, 0, False
            # 환경 초기화 및 초기상태관측 
            state = self.env.reset()
            while not done:
                # 환경 가사화
                self.env.render()
                # 행동 추출
                action = self.actor.get_action(state)
                # 행동 범위 클리핑
                action = np.clip(action, -self.action_bound, self.action_bound)
                # 다음 상태, 보상 관즉
                next_state, reward, done, _ = self.env.step(action)
                # shape 변환
                state = np.reshape(state,[1,self.state_dim])
                next_state = np.reshape (next_state, [1,self.state_dim])
                action = np.reshape(action, [1 ,self.action_dim])
                reward = np.reshape(reward, [1,1])
                # 상태가치 계산
                v_value = self.critic.model.predict(state)
                next_v_value = self.critic.model.predict(next_state)
                # 어드밴티지와 TD 타깃 계산 
                train_reward = (reward+8)/8
                advantage, y_i = self.advantage_td_target(train_reward, v_value, next_v_value, done)
                # 배치에 저장
                batch_state.append(state)
                batch_action.append(action)
                batch_td_target.append(y_i)
                batch_advantage.append(advantage)
                
                # 배치가 채워질 때까지 학습하지 않고 저장만 계속 
                if len(batch_state) < self.BATCH_SIZE:
                    # 상태 업데이트 
                    state = next_state[0]
                    episode_reward += reward[0]
                    time += 1
                    continue
                # 배치가 채워지면 학습 진행
                # 배치에서 데이터 추출
                states = self.unpack_batch(batch_state)
                actions = self.unpack_batch(batch_action)
                td_targets = self.unpack_batch(batch_td_target)
                advantages = self.unpack_batch(batch_advantage)
                # 배치 비움
                batch_state, batch_action, batch_td_target, batch_advantage = [], [], [], []
                
                # 크리틱 신경망 업데이트 
                self.critic.train_on_batch(states, td_targets)
                # 액터 신경망 업데이트
                self.actor.train(states, actions, advantages)
                
                # 상태 업데이트
                state = next_state[0]
                episode_reward += reward[0]
                time += 1
            
            # 에피소드마다 결과 보상값 출력
            print( 'Episode: ' , ep+1, 'Ti me: ' , time, 'Reward: ' , episode_reward) 
            self.save_epi_reward.append(episode_reward)
            
            # 에피소드 10 번마다 신경망 파라미터를 파일에 저장 
            if ep%10 == 0:
                self.actor.save_weights( "./save_weights/pendulum_actor.h5") 
                self.critic.save_weights( "./save_weights/pendulum_critic.h5")
        # 학습이 끝난 후, 누적 보상값 저장 
        np.savetxt( './save_weights/pendulum_epi_reward.txt', self. save_epi_reward)
    
    ## 에피소드와 누적 보상값을 그려주는 항수 
    def plot_result(self):
        plt.plot(self.save_epi_reward)
        plt.show() 