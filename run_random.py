import numpy as np
import sys
from uofgsocsai import LochLomondEnv
from helpers import *
from collections import namedtuple

def main(problem_id, map_name_base): 
    #random agent derived from lochlomond_demo.py provided by tutor prof.bjorn jensen for ai course(2019-20) University of Glasgow
    if(problem_id < 0 or problem_id > 7):
        problem_id = problem_id
    else:
        print("Probleam ID should be between 0 and 7")
    
    if(map_name_base == "8x8-base" or map_name_base == "4x4-base"):
        map_name_base = map_name_base 
    else:
        print("Map base can be 8x8-base or 4x4-base")
    
    reward_hole = 0.0     
    is_stochastic = True  
    EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards"])
    max_episodes = 10000  
    max_iter_per_episode = 1000 
    
    #generate the specific problem
    env = LochLomondEnv(problem_id=problem_id, is_stochastic=is_stochastic, map_name_base=map_name_base, reward_hole=reward_hole)

    env.action_space.sample() 


    print(env.desc)

    state_space_locations, state_space_actions, state_initial_id, state_goal_id = env2statespace(env)

   
    np.random.seed(12)
    stats = EpisodeStats(episode_lengths=np.zeros(max_episodes),episode_rewards=np.zeros(max_episodes))

    for e in range(max_episodes): 
        observation = env.reset()      

        for iter in range(max_iter_per_episode):      
          action = env.action_space.sample() #The agent goes here
          observation, reward, done, info = env.step(action) 

          stats.episode_rewards[e] += reward #collect useful stats for comparison and plotting
          stats.episode_lengths[e] = iter
          
          if(done and reward==reward_hole): 
              print("We have reached a hole :-( [we can't move so stop trying; just give up... and perhaps restart]")
              break

          if (done and reward == +1.0):
              #env.render()     
              print("We have reached the goal :-) [stop trying to move; we can't]. That's ok we have achived the goal... perhaps try again?]")
              break

    return (stats)

if __name__ == "__main__":
   main(sys.argv[1])
#main(1, "4x4-base")



