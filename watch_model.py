from custom_gym.doublecartpole import DoubleCartPoleEnv
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np

OBSERVATION_SPACE_DIMS = 6
ACTION_SPACE = [0,1]
ALPHA = 0.001

def create_dqn(action_space, observation_space):
    nn = Sequential()
    nn.add(Dense(400, input_dim=OBSERVATION_SPACE_DIMS, activation="relu"))
    nn.add(Dense(200, activation='relu'))
    nn.add(Dense(100, activation='relu'))
    nn.add(Dense(100, activation='relu'))
    nn.add(Dense(len(ACTION_SPACE), activation='linear'))
    nn.compile(loss='mse', optimizer=Adam(lr=ALPHA))
    return nn
        
                  
class DoubleDQNAgent(object):

       
    def __init__(self, action_space, observation_space):
        self.action_space = action_space
        self.observation_space = observation_space
        self.online_network = create_dqn(action_space, observation_space)
        self.target_network = create_dqn(action_space, observation_space)

    
    
    def act(self, state):
        state = self._reshape_state_for_net(state)
        q_values = self.online_network.predict(state)[0]
        return np.argmax(q_values)

    def _reshape_state_for_net(self, state):
        return np.reshape(state,(1, OBSERVATION_SPACE_DIMS))  

    def load_model(self):
        try:
            self.target_network.load_weights('./model/weights_target')
            self.online_network.load_weights('./model/weights_online')
        except:
            pass   

def test_agent():
    env = DoubleCartPoleEnv()
    trials = []
    MAX_STEPS_PER_EPISODE = 10000

    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    log_list = list()
    agent = DoubleDQNAgent(action_space, observation_space)
    agent.load_model()

    while 1:
        state = env.reset()
        episode_score = 0
        #steps =0
        for _ in range(MAX_STEPS_PER_EPISODE):
            action = agent.act(state)
            
            next_state, reward, done, _ = env.step(action)
            env.render()
            episode_score += reward
            #steps+=1
            state = next_state
            if done:
                break


if __name__ == '__main__':
    test_agent()