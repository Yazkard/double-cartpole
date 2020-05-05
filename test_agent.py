import gym
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
import random
from matplotlib import pyplot as plt
from custom_gym.doublecartpole import DoubleCartPoleEnv


# AGENT/NETWORK HYPERPARAMETERS
EPSILON_INITIAL = 1.0 # exploration rate
EPSILON_DECAY = 0.9997
EPSILON_MIN = 0.01
ALPHA = 0.001 # learning rate
GAMMA = 0.98 # discount factor
TAU = 0.1 # target network soft update hyperparameter
EXPERIENCE_REPLAY_BATCH_SIZE = 300
AGENT_MEMORY_LIMIT = 6000
MIN_MEMORY_FOR_EXPERIENCE_REPLAY = 500
STEPS_BEFORE_REPLAY = 300
TARGET_UPDATE_RATIO = 5

OBSERVATION_SPACE_DIMS = 6
ACTION_SPACE = [0,1]



def create_dqn(action_space, observation_space):
    nn = Sequential()
    nn.add(Dense(400, input_dim=OBSERVATION_SPACE_DIMS, activation="relu"))
    nn.add(Dense(200, activation='relu'))
    nn.add(Dense(100, activation='relu'))
    nn.add(Dense(50, activation='relu'))
    nn.add(Dense(len(ACTION_SPACE), activation='linear'))
    nn.compile(loss='mse', optimizer=Adam(lr=ALPHA))
    return nn

def takeScore(elem):
    return elem[5]


def choose_best_steps(memory, n):
    best = list()
    best = memory[0:n]
    best.sort(key= takeScore, reverse=True)
    for elem in memory:
        if elem[5] > best[n-1][5]:
            best[n-1] = elem
            best.sort(key= takeScore, reverse=True)

    return best



                  
                  
class DoubleDQNAgent(object):

       
    def __init__(self, action_space, observation_space):
        self.memory = []
        self.action_space = action_space
        self.observation_space = observation_space
        self.online_network = create_dqn(action_space, observation_space)
        self.target_network = create_dqn(action_space, observation_space)
        self.epsilon = EPSILON_INITIAL
        self.has_talked = False
    
    
    def act(self, state):
        if self.epsilon > np.random.rand():
            # explore
            return np.random.choice(self.action_space)
        else:
            # exploit
            state = self._reshape_state_for_net(state)
            q_values = self.online_network.predict(state)[0]
            return np.argmax(q_values)


    def experience_replay(self):

        minibatch = random.sample(self.memory, EXPERIENCE_REPLAY_BATCH_SIZE)
        #minibatch_best = choose_best_steps(self.memory, EXPERIENCE_REPLAY_BATCH_SIZE)
        #minibatch.extend(minibatch_best)
        minibatch_new_q_values = []

        for state, action, reward, next_state, done in minibatch:
            state = self._reshape_state_for_net(state)
            experience_new_q_values = self.online_network.predict(state)[0]
            if done:
                q_update = reward
            else:
                next_state = self._reshape_state_for_net(next_state)
                # using online network to SELECT action
                online_net_selected_action = np.argmax(self.online_network.predict(next_state))
                # using target network to EVALUATE action
                target_net_evaluated_q_value = self.target_network.predict(next_state)[0][online_net_selected_action]
                q_update = reward + GAMMA * target_net_evaluated_q_value
            experience_new_q_values[action] = q_update
            minibatch_new_q_values.append(experience_new_q_values)
        minibatch_states = np.array([state for state,_,_,_,_ in minibatch])
        minibatch_new_q_values = np.array(minibatch_new_q_values)
        self.online_network.fit(minibatch_states, minibatch_new_q_values, verbose=False, epochs=1)
        
        
    def update_target_network(self):
        q_network_theta = self.online_network.get_weights()
        target_network_theta = self.target_network.get_weights()
        counter = 0
        for q_weight, target_weight in zip(q_network_theta,target_network_theta):
            target_weight = target_weight * (1-TAU) + q_weight * TAU
            target_network_theta[counter] = target_weight
            counter += 1
        self.target_network.set_weights(target_network_theta)


    def remember(self, state, action, reward, next_state, done):
        if len(self.memory) > AGENT_MEMORY_LIMIT:
            self.memory.pop(0)
        experience = (state, action, reward, next_state, done)
        self.memory.append(experience)
                  
                  
    def update_epsilon(self):
        self.epsilon = max(self.epsilon * EPSILON_DECAY, EPSILON_MIN)

    def _reshape_state_for_net(self, state):
        return np.reshape(state,(1, OBSERVATION_SPACE_DIMS))  

    def save_model(self):
        self.online_network.save_weights('./model/weights')

    def load_model(self):
        try:
            self.target_network.load_weights('./model/weights')
            self.online_network.load_weights('./model/weights')
        except:
            pass   


def test_agent():
    env = DoubleCartPoleEnv()
    trials = []
    NUMBER_OF_TRIALS=1
    MAX_TRAINING_EPISODES = 1000000
    MAX_STEPS_PER_EPISODE = 20000

    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n

    for trial_index in range(NUMBER_OF_TRIALS):
        agent = DoubleDQNAgent(action_space, observation_space)
        agent.load_model()
        trial_episode_scores = []
        s = 0

        for episode_index in range(1, MAX_TRAINING_EPISODES+1):
            state = env.reset()
            episode_score = 0
            steps =0
            

            for _ in range(MAX_STEPS_PER_EPISODE):
                action = agent.act(state)
                
                next_state, reward, done, _ = env.step(action)
                #env.render()
                episode_score += reward
                s+=1
                steps+=1
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                if s > STEPS_BEFORE_REPLAY:
                    agent.experience_replay()
                    agent.update_target_network()
                    agent.save_model()
                    s=0
                
                if done:
                    break
            
            trial_episode_scores.append(episode_score)
            agent.update_epsilon()
            last_100_avg = np.mean(trial_episode_scores[-100:])
            

            with open("log3.log", "a") as myfile:
                myfile.write("Run: " + str(episode_index) + ", steps_done: " + str(steps) + ", avg_points_per_step: " + str(episode_score/steps) + ", exploration: " + str(agent.epsilon) + ", score: " + str(episode_score) +", avg_last_100_score: " + str(last_100_avg)+"\n")


  
            #print("Run: " + str(episode_index) + ", exploration: " + str(agent.epsilon) + ", score: " + str(episode_score) +", score: " + str(last_100_avg), file=sample)
        
            
            #print 'E %d scored %d, avg %.2f' % (episode_index, episode_score, last_100_avg)
            #if len(trial_episode_scores) >= 100 and last_100_avg >= 195.0:
            #    print 'Trial %d solved in %d episodes!' % (trial_index, (episode_index - 100))
            #    break
        trials.append(np.array(trial_episode_scores))
    return np.array(trials)



def plot_trials(trials):
    _, axis = plt.subplots()    

    for i, trial in enumerate(trials):
        steps_till_solve = trial.shape[0]-100
        # stop trials at 2000 steps
        if steps_till_solve < 1900:
            bar_color = 'b'
            bar_label = steps_till_solve
        else:
            bar_color = 'r'
            bar_label = 'Stopped at 2000'
        plt.bar(np.arange(i,i+1), steps_till_solve, 0.5, color=bar_color, align='center', alpha=0.5)
        axis.text(i-.25, steps_till_solve + 20, bar_label, color=bar_color)

    plt.ylabel('Episodes Till Solve')
    plt.xlabel('Trial')
    trial_labels = [str(i+1) for i in range(len(trials))]
    plt.xticks(np.arange(len(trials)), trial_labels)
    # remove y axis labels and ticks
    axis.yaxis.set_major_formatter(plt.NullFormatter())
    plt.tick_params(axis='both', left='off')

    plt.title('Double DQN CartPole v-0 Trials')
    plt.show()


def plot_individual_trial(trial):
    plt.plot(trial)
    plt.ylabel('points in Episode')
    plt.xlabel('Episode')
    plt.title('Double DQN points in Select Trial')
    plt.show()


if __name__ == '__main__':
    trials = test_agent()
    # print 'Saving', file_name
    # np.save('double_dqn_cartpole_trials.npy', trials)
    # trials = np.load('double_dqn_cartpole_trials.npy')
    #plot_trials(trials)
    #plot_individual_trial(trials[1])