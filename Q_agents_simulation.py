import numpy as np
import collections

from numpy.core.fromnumeric import argmax
import flappy_bird_gym
import time
import math
import json


def obs2state(obs, multiplier=1000):
        x_pos = int(math.floor(obs[0]*multiplier))
        y_pos = int(math.floor(obs[1]*multiplier))
        y_vel = int(obs[2])
        state_string = str(x_pos) + '_' + str(y_pos) + '_' + str(y_vel)
        #state_string = str(x_pos) + '_' + str(y_pos)

        return state_string

def test(Q,iteration):
    Q = collections.defaultdict(list,Q)
    env = flappy_bird_gym.make("FlappyBird-cust-v0")
    env.observation_space.sample
    max_score = 0
    for i in range(iteration):
        obs = env.reset()
        state = obs2state(obs)
        done = False
        
        traject = []
        while not done:
            if len(Q[state]) == 0:
                action = env.action_space.sample()      
            else:
                action = np.argmax(Q[state])
        
            next_obs, reward, done, info = env.step(action) #two cases, reward always 1, reward always 0
            next_state = obs2state(next_obs)
            state = next_state
            traject.append(action)

            """
            This line is the logic of which result you want to achieve.
            """
            if info['score'] > 1000 and done:
                print('Achieved score: ', info['score'])
                return traject
        print(i)
    env.close()
    return traject

def simulation(traject):
    env = flappy_bird_gym.make("FlappyBird-replay-v0")
    obs = env.reset()
    for i in range(len(traject)):
        reward, done, info = env.step(traject[i])
        env.render()
        time.sleep(1/60)
    env.close()




if __name__ == "__main__":
    with open("Q.json") as f:
        Q2 = json.load(f)
    traject = test(Q2,10000)
    simulation(traject)
    # with open('trajectory.npy', 'wb') as f:
    #     np.save(f, np.array(traject))

