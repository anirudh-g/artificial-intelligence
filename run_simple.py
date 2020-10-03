
import numpy as np
from uofgsocsai import LochLomondEnv
import os, sys
from helpers import *
from collections import namedtuple

import networkx as nx
print("networkx version:"+nx.__version__)

AIMA_TOOLBOX_ROOT="aima-python-uofg_v20192020b"
sys.path.append(AIMA_TOOLBOX_ROOT)

from search import *
import warnings
warnings.filterwarnings("ignore")
print("Working dir:"+os.getcwd())
print("Python version:"+sys.version)



def get_action_from_states(cur_node, next_node):
    # Action to int representations (taken from the FrozenLake github page)
    # (https://github.com/openai/gym/blob/master/gym/envs/toy_text/frozen_lake.py)
    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3

    # Get the coordinates from the state string for each node
    x1 = cur_node.state[2]
    y1 = cur_node.state[4]
    x2 = next_node.state[2]
    y2 = next_node.state[4]
    
    # We need to account for rotation between graph and environment
    # X on our graph became Y on our environment (handles up/down)
    # Y on our graph became X on our environment (handles left/right)
    if x1 == x2:
        if y1 > y2:
            return LEFT
        else:
            return RIGHT
    else:
        if x1 > x2:
            return UP
        else:
            return DOWN

def my_best_first_graph_search_for_vis(problem, f):
    """Search the nodes with the lowest f scores first.
    You specify the function f(node) that you want to minimize; for example,
    if f is a heuristic estimate to the goal, then we have greedy best
    first search; if f is node.depth then we have breadth-first search.
    There is a subtlety: the line "f = memoize(f, 'f')" means that the f
    values will be cached on the nodes as they are computed. So after doing
    a best first search you can examine the f values of the path returned."""
    
    # we use these two variables at the time of visualisations
    iterations = 0
    all_node_colors = []
    node_colors = {k : 'white' for k in problem.graph.nodes()}
    
    f = memoize(f, 'f')
    node = Node(problem.initial)
    
    node_colors[node.state] = "red"
    iterations += 1
    all_node_colors.append(dict(node_colors))
    
    if problem.goal_test(node.state):
        node_colors[node.state] = "green"
        iterations += 1
        all_node_colors.append(dict(node_colors))
        return(iterations, all_node_colors, node)
    
    frontier = PriorityQueue('min', f)
    frontier.append(node)
    
    node_colors[node.state] = "orange"
    iterations += 1
    all_node_colors.append(dict(node_colors))
    
    explored = set()
    while frontier:
        node = frontier.pop()
        
        node_colors[node.state] = "red"
        iterations += 1
        all_node_colors.append(dict(node_colors))
        
        if problem.goal_test(node.state):
            node_colors[node.state] = "green"
            iterations += 1
            all_node_colors.append(dict(node_colors))
            return(iterations, all_node_colors, node)
        
        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                frontier.append(child)
                node_colors[child.state] = "orange"
                iterations += 1
                all_node_colors.append(dict(node_colors))
            elif child in frontier:
                incumbent = frontier[child]
                if f(child) < f(incumbent):
                    del frontier[incumbent]
                    frontier.append(child)
                    node_colors[child.state] = "orange"
                    iterations += 1
                    all_node_colors.append(dict(node_colors))

        node_colors[node.state] = "gray"
        iterations += 1
        all_node_colors.append(dict(node_colors))
    return None


def my_astar_search_graph(problem, h=None):
    """A* search is best-first graph search with f(n) = g(n)+h(n).
    You need to specify the h function when you call astar_search, or
    else in your Problem subclass."""
    h = memoize(h or problem.h, 'h')
    iterations, all_node_colors, node = my_best_first_graph_search_for_vis(problem, 
                                                                lambda n: n.path_cost + h(n))
    return(iterations, all_node_colors, node)


def main(problem_id, map_name_base): 
    #simple agent referenced and adapted from lab 4 notebook by tutor prof.bjorn jensen for ai course (2019-20)
    if(problem_id < 0 or problem_id > 7):
        problem_id = problem_id
    else:
        print("Probleam ID should be between 0 and 7")
    
    if(map_name_base == "8x8-base" or map_name_base == "4x4-base"):
        map_name_base = map_name_base 
    else:
        print("Map base can be 8x8-base or 4x4-base")
    
    reward_hole = -1.0     
    is_stochastic = False  

    max_episodes = 10000 

    env = LochLomondEnv(problem_id=problem_id, is_stochastic=is_stochastic, map_name_base=map_name_base, reward_hole=reward_hole)

    env.action_space.sample() 

    print(env.desc)
    EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards"])
    state_space_locations, state_space_actions, state_initial_id, state_goal_id = env2statespace(env)

    frozen_lake_map = UndirectedGraph(state_space_actions)
    frozen_lake_map.locations = state_space_locations
    frozen_lake_problem = GraphProblem(state_initial_id, state_goal_id, frozen_lake_map)

    all_node_colors=[]
    iterations, all_node_colors, node = my_astar_search_graph(problem=frozen_lake_problem, h=None)

    solution_path = [node]
    cnode = node.parent
    solution_path.append(cnode)
    while cnode.state != "S_00_00":    
        cnode = cnode.parent
        if cnode is None:
            break
        solution_path.append(cnode)


    steps = solution_path[::-1]
    # Reset the random generator to a known state (for reproducibility)
    np.random.seed(12)
    
    observation = env.reset() # reset the state of the env to the starting state     

    stats = EpisodeStats(episode_lengths=np.zeros(max_episodes),episode_rewards=np.ones(max_episodes))
    for e in range(max_episodes): # iterate over episodes

        observation = env.reset() # reset the state of the env to the starting state     

        for i in range(len(steps)-1):
            action =  get_action_from_states(steps[i],steps[i+1])# your agent goes here (the current agent takes random actions)
            
            observation, reward, done, info = env.step(action) # observe what happends when you take the action
            # update stats
            stats.episode_rewards[e] = reward
            stats.episode_lengths[e] = i
    
          # Check if we are done and monitor rewards etc...
        if (done):
        
            print("We have reached the goal :-) [stop trying to move; we can't]. That's ok we have achived the goal... perhaps try again?]")
            break

    return (stats)

#main(0, "8x8-base")
if __name__ == "__main__":
   main(sys.argv[1])