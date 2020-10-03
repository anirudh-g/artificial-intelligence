import sys
import random
import numpy as np
from collections import namedtuple
from uofgsocsai import LochLomondEnv # load the class defining the custom Open AI Gym problem

def main(problem_id, map_name_base):
    #rl agent referenced from lab 8 and 9 notebooks provided by tutor prof.bjorn jensen for ai course(2019-20) University of Glasgow
    if(problem_id < 0 or problem_id > 7):
        problem_id = problem_id
    else:
        print("Problem ID should be between 0 and 7")
    
    if(map_name_base == "8x8-base" or map_name_base == "4x4-base"):
        map_name_base = map_name_base 
    else:
        print("Map base can be 8x8-base or 4x4-base")
    
    reward_hole = -0.05 #Hole penalty is set based on analysis to ensure that the reward is maximized
    is_stochastic = True 
    EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards"])
    map_name_base=map_name_base

    np.random.seed(12)
    env = LochLomondEnv(problem_id=problem_id, is_stochastic=is_stochastic, reward_hole=reward_hole, map_name_base=map_name_base)

    states = env.observation_space.n
    actions = env.action_space.n
    Q = np.zeros((states, actions))

    max_episodes = 10000  
    max_iter_per_episode = 1000

    alpha = 0.1  #learning rate
    gamma = 0.999 #discount rate
    epsilon = 1  
    stats = EpisodeStats(episode_lengths=np.zeros(max_episodes),episode_rewards=np.zeros(max_episodes))

    for episode in range(max_episodes):
        state = env.reset()
        
        for step in range(max_iter_per_episode):
            # take best action according to Q-table if random value is greater than epsilon, otherwise take a random action
            random_value = random.uniform(0,1)
            if random_value > epsilon:
                action = np.argmax(Q[state,:]) #Agent goes here
            else:
                action = env.action_space.sample()

            new_state, reward, done, info = env.step(action)
            Q[state, action] = Q[state,action] + alpha * (reward + gamma*np.max(Q[new_state,:]) - Q[state,action])
            stats.episode_rewards[episode] += reward
            stats.episode_lengths[episode] = step
            state = new_state

            if done:
                    break

        epsilon = 0.01 #epsilon is set to a low value to make sure of the exploitation
    
    print(Q)
    
    return(stats)

if __name__ == "__main__":
   main(sys.argv[1])
