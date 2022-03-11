from keras.models import Model 
from keras.layers import Dense, Input 
from keras.optimizers import Adam 

class Critic(object) : 
    ##A2C 의 크리틱 신경망 
    def __init__(self, state_dim, action_dim, learning_rate):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        
        # 크리틱 신경망 생성 
        self.model , self.states = self.build_network()
        
        # 학습 방법 설정 
        self.model.compile(optimizer=Adam(self.learning_rate), loss='mse')
    ## 크리틱 신경망 
    def build_network(self):
        state_input = Input((self.state_dim,))
        h1 = Dense(64, activation= 'relu')(state_input)
        h2 = Dense(32, activation='relu')(h1)
        h3 = Dense(16, activation='relu')(h2)
        v_output = Dense(1,activation= 'linear')(h3)
        model = Model(state_input, v_output)
        model.summary()
        return model,state_input 
    ## 배치 데이터 (batch data) 로 크리틱 신경망 업데이트 
    def train_on_batch(self, states, td_targets):
        return self.model .train_on_batch(states, td_targets) 
    
    ## 크리틱 신경망 파라미터 저장
    def save_weights(self, path): 
        self.model.save_weights(path)
    ## 크리틱 신경망 파라미터 로드 
    def load_weights(self, path):
        self.model.load_weights(path + 'pendulum_critic.h5') 
        