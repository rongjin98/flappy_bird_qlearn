import numpy as np
import collections
import flappy_bird_gym
import time
import math
import json

class QLearning:
    def __init__(self,MaxEpisode = 10000) :
        self.MaxEposide = MaxEpisode
        self.reward = {0:1, 1:-1000} #idk, maybe useless
        self.action = [0, 1]
        self.Q = collections.defaultdict(lambda: tuple([0,0]))
        self.state_cnt = collections.Counter()
        self.env = flappy_bird_gym.make("FlappyBird-v0")
        #self.env.observation_space.sample


    def obs2state(obs, multiplier=1000):
        x_pos = int(math.floor(obs[0]*multiplier))
        y_pos = int(math.floor(obs[1]*multiplier))
        y_vel = int(obs[2])
        #y_dist = obs[3]
        state_string = str(x_pos) + '_' + str(y_pos) + '_' + str(y_vel)# + '_' + str(y_dist)
        #state_string = str(x_pos) + '_' + str(y_pos)

        return state_string

    def train(self, dumpInterval = 100, evalInterval = 100, epsilon = 0.2, lr = 0.8, decay = 0.8):
        """
        Several parameters we want to record:
        1. Max score the bird ever achieved during training
        2. The total sum of score per 10,000 episodes to reflect the tendency of return
        3. Keep track on previous total score of 10k episodes, so that we only dump the Q-value at the maximum return
           -- previous total score is no longer used since training speed slows down quickly as the learning progress
        
        Note: Once reach 340k episodes, every 10k episodes take aound 8 hours to complete, and the gain of total return decay significantly

        """
        
        max_score = 0
        interval_score_sum = [0, 1] #default value
        total_score = 0
        #previous_total_score = -999 #default value
        episodes = 0
        while episodes <self.MaxEposide: #and (interval_score_sum[-1] > interval_score_sum[-2]):
            obs = self.env.reset()
            state = QLearning.obs2state(obs)
            traject_recorder = [] #track the whole trajectory history until done
            done = False
            
            prev_score = 0
            if episodes % evalInterval == evalInterval-1:
                interval_score_sum.append(total_score)
                # if total_score > previous_total_score:
                #     previous_total_score = total_score
                total_score = 0
                print(episodes)
                print(interval_score_sum)

            while True: 
                epsilon_decay = 0.99**(self.state_cnt[state]) #epslion_decay changed by state cnt
                self.state_count_update(state)

                if np.random.uniform(0, 1) < epsilon*epsilon_decay:
                    action = self.env.action_space.sample()
                else: 
                    action = np.argmax(self.Q[state])
                
                

                next_obs, reward, done, info = self.env.step(action)
                next_state = QLearning.obs2state(next_obs)
                score = info['score']

                reward = 0
                if prev_score < score: #if score changed, reward it with 10 points
                    prev_score = score
                    reward = 10
                    total_score += 1
            
                if max_score < score: #record the max score achieved in single episode
                    max_score = score
                    episode_at_maximum = episodes
                
                if episodes % evalInterval == 0: #render every 10k episodes
                    self.env.render()
                    time.sleep(1/60)

                if done:
                    hit_dist = int(next_state.split('_')[1]) #high dist flag punishment
                    if hit_dist > 120:
                        reward = -10*hit_dist
                        if len(traject_recorder) < 3: # Additional punishment if the bird dies too soon
                            reward += -1000
                    else:
                        reward = -500 #regular punishment if done
                    traject_recorder.append([state, action, reward, next_state]) 
                    traject_recorder = reversed(traject_recorder) #reverse the trajectory for backward propagation
                    #backward propagation update information much faster than forward propagation

                    QLearning.Q_value_update(self,traject_recorder, lr, decay)
                    if episodes % dumpInterval == 0: #dump Q periodically
                        QLearning.dump_Q_json(self)
                    break

                traject_recorder.append([state, action, reward, next_state])
                state = next_state
            episodes +=1
        
        interval_score_sum = np.array(interval_score_sum)
        QLearning.dump_return_every_iteration(interval_score_sum)#save the return tendency of every 10k episodes for plotting
        print(len(self.Q)) # The length of Q reflect the size of statespace the bird explored
        print('The maximum score is: ', max_score, 'at Itertaion: ', episode_at_maximum)

        # if total_score > previous_total_score:
        #     print(previous_total_score)
        #     QLearning.dump_Q_json(self)
        self.env.close()
    
    def Q_value_update(self, traject_, lr, decay):
        for transition_set in traject_:
            state = transition_set[0]
            action = transition_set[1]
            reward = transition_set[2]
            next_state = transition_set[3]
            action_set = list(self.Q[state])
            action_set[action] = (1-lr)*self.Q[state][action] +lr*(reward \
                + decay*np.max(self.Q[next_state]))
            self.Q[state] = tuple(action_set)

    def state_count_update(self, state):
        self.state_cnt[state] += 1

    def dump_Q_json(self):
        with open('Q.json','w') as f:
            json.dump(self.Q,f)
    
    def dump_return_every_iteration(score_sum):
        with open('gainIter.npy', 'wb') as f:
            np.save(f, score_sum)

if __name__ == "__main__":
    Q_learn = QLearning(34*10**4) #the recommended total training episodes
    Q_learn.train(10000,10000)
