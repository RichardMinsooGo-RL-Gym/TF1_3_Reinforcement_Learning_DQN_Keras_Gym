import sys
import gym
import pylab
import random
import numpy as np
import os
import time, datetime
from collections import deque

from keras.layers import *
from keras.models import Sequential,Model
import keras
from keras import backend as K_back
from keras.optimizers import Adam

# In case of CartPole-v1, maximum length of episode is 500
env = gym.make('CartPole-v1')
# env.seed(1)     # reproducible, general Policy gradient has high variance
# env = env.unwrapped

# get size of state and action from environment
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

file_name =  sys.argv[0][:-3]

model_path = "save_model/" + file_name
graph_path = "save_graph/" + file_name

if not os.path.isdir(model_path):
    os.mkdir(model_path)

if not os.path.isdir(graph_path):
    os.mkdir(graph_path)

# DQN Agent for the Cartpole
# it uses Neural Network to approximate q function
# and replay memory & target q network
class DQN:
    def __init__(self, state_size, action_size):
        # if you want to see Cartpole learning, then change to True
        self.render = False
        self.load_model = False

        # get size of state and action
        self.progress = " "
        self.action_size = action_size
        self.state_size = state_size
        
        # train time define
        self.training_time = 15*60
        
        # These are hyper parameters for the DQN
        self.learning_rate = 0.001
        self.discount_factor = 0.99
        
        self.epsilon_max = 1.0
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.999
        self.epsilon_rate = self.epsilon_max
        
        self.hidden1, self.hidden2 = 24, 24
        
        self.ep_trial_step = 500
        
        # Parameter for Experience Replay
        self.size_replay_memory = 2000
        self.batch_size = 64
        
        # Experience Replay 
        self.memory = deque(maxlen=self.size_replay_memory)
        
        # Parameter for Target Network
        self.target_update_cycle = 200

        # create main model and target model
        self.model = self.build_model()
        self.target_model = self.build_model()

        # initialize target model
        self.Copy_Weights()

        if self.load_model:
            self.model.load_weights(model_path + "/model.h5")

    # approximate Q function using Neural Network
    # state is input and Q Value of each action is output of network
    def build_model(self):
        
        state = Input(shape=(self.state_size,))
        x = Dense(units=32,activation='relu',kernel_initializer='he_uniform',name='hidden_layer_1')(state)
        x = Dense(units=32,activation='relu',kernel_initializer='he_uniform',name='hidden_layer_2')(x)
        value_= Dense(units=1,activation='linear',kernel_initializer='he_uniform',name='Value_func')(x)
        ac_activation = Dense(units=self.action_size,activation='linear',kernel_initializer='he_uniform',name='action')(x)
        
        #Compute average of advantage function
        avg_ac_activation = Lambda(lambda x: K_back.mean(x,axis=1,keepdims=True))(ac_activation)
        
        #Concatenate value function to add it to the advantage function
        concat_value = Concatenate(axis=-1,name='concat_0')([value_,value_])
        concat_avg_ac = Concatenate(axis=-1,name='concat_ac_{}'.format(0))([avg_ac_activation,avg_ac_activation])

        for i in range(1,self.action_size-1):
            concat_value = Concatenate(axis=-1,name='concat_{}'.format(i))([concat_value,value_])
            concat_avg_ac = Concatenate(axis=-1,name='concat_ac_{}'.format(i))([concat_avg_ac,avg_ac_activation])

        #Subtract concatenated average advantage tensor with original advantage function
        ac_activation = Subtract()([ac_activation,concat_avg_ac])
        
        #Add the two (Value Function and modified advantage function)
        merged_layers = Add(name='final_layer')([concat_value,ac_activation])
        model = Model(inputs = state,outputs=merged_layers)
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    # after some time interval update the target model to be same with model
    def Copy_Weights(self):
        self.target_model.set_weights(self.model.get_weights())

    # get action from model using epsilon-greedy policy
    def get_action(self, state):
        #Exploration vs Exploitation
        if np.random.rand() <= self.epsilon_rate:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(state)
            return np.argmax(q_value[0])

    # save sample <s,a,r,s'> to the replay memory
    def append_sample(self, state, action, reward, next_state, done):
        #in every action put in the memory
        self.memory.append((state, action, reward, next_state, done))

    def get_target_q_value(self, next_state, reward):
        
        # max Q value among next state's actions
        # DQN chooses the max Q value among next actions
        # selection and evaluation of action is 
        # on the target Q Network
        # Q_max = max_a' Q_target(s', a')
        q_value = np.amax(self.target_model.predict(next_state)[0])

        # Q_max = reward + discount_factor * Q_max
        q_value *= self.discount_factor
        q_value += reward
        return q_value

    def train_model(self):
        
        # sars = state, action, reward, state' (next_state)
        minibatch = random.sample(self.memory, self.batch_size)
        states, q_values_batch = [], []

        # fixme: for speedup, this could be done on the tensor level
        # but easier to understand using a loop
        for state, action, reward, next_state, done in minibatch:
            # policy prediction for a given state
            q_values = self.model.predict(state)
            
            # get Q_max
            q_value = self.get_target_q_value(next_state, reward)

            # correction on the Q value for the action used
            q_values[0][action] = reward if done else q_value

            # collect batch state-q_value mapping
            states.append(state[0])
            q_values_batch.append(q_values[0])

        # train the Q-network
        self.model.fit(np.array(states), np.array(q_values_batch), batch_size=self.batch_size, epochs=1,verbose=0)
        
        if self.epsilon_rate > self.epsilon_min:
            self.epsilon_rate *= self.epsilon_decay
        
def main():
    
    # DQN 에이전트의 생성
    agent = DQN(state_size, action_size)
    
    last_n_game_reward = deque(maxlen=20)
    last_n_game_reward.append(0)
    avg_ep_step = np.mean(last_n_game_reward)
    
    display_time = datetime.datetime.now()
    print("\n\n Game start at :",display_time)
    
    start_time = time.time()
    agent.episode = 0
    time_step = 0
    
    while time.time() - start_time < agent.training_time and avg_ep_step < 495:
        
        done = False
        ep_step = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        while not done and ep_step < agent.ep_trial_step:
            if len(agent.memory) < agent.size_replay_memory:
                agent.progress = "Exploration"
            else :
                agent.progress = "Training"

            ep_step += 1
            time_step += 1
            
            if agent.render:
                env.render()
                
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            
            next_state = np.reshape(next_state, [1, state_size])
            agent.append_sample(state, action, reward, next_state, done)
            state = next_state
            
            if agent.progress == "Training":
                agent.train_model()
                # if done or ep_step % agent.target_update_cycle == 0:
                if done or ep_step % agent.target_update_cycle == 0:
                    # return# copy q_net --> target_net
                    agent.Copy_Weights()

            if done or ep_step == agent.ep_trial_step:
                if agent.progress == "Training":
                    agent.episode += 1
                    last_n_game_reward.append(ep_step)
                    avg_ep_step = np.mean(last_n_game_reward)
                print("episode :{:>5d} / ep_step :{:>5d} / last 20 game avg :{:>4.1f}".format(agent.episode, ep_step, avg_ep_step))
                break
                
    agent.model.save_weights(model_path + "/model.h5")
    
    e = int(time.time() - start_time)
    print(' Elasped time :{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60))
    sys.exit()
                    
if __name__ == "__main__":
    main()
