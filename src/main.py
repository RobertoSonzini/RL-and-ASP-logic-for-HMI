import matplotlib.pyplot as plt
import copy
import json
import os
import numpy as np
import itertools
import random

from environment_solver import *
from human import Human
from environment import SocialEnvironment
from agent import Agent
from config import *
from plots import *
from training import *

def simulate_dynamic_session(num_envs = 5):
    human = Human(settings.get_min_value(), settings.get_max_value())
    agent = Agent(settings.get_min_value(), settings.get_max_value(), settings.get_step_size())

    results = {'history': {'happiness': [], 'sadness': [], 'anger': [], 'anxiety': []}, 
               'actions': [],
               'explored_states': []}
    
    step = 0

    widths = [random.randint(settings.get_min_window_width(), settings.get_max_window_width()) for _ in range(num_envs)]

    windows = [0]
    total = 0
    for w in widths:
        total += w
        windows.append(total)

    envs = [sample_environment(ENV_OBS) for _ in range(num_envs)]

    obs = [observations_to_facts(generate_observations(env, ENV_OBS)) for env in envs]

    print("Step intervals:", [(windows[i], windows[i+1]) for i in range(len(windows)-1)])
    
    for i in range(len(windows)-1):
        start, end = windows[i], windows[i+1]
        env_name = envs[i]
        facts = obs[i]

        environment = SocialEnvironment(context_type=env_name)

        inferred_env = infer_environment("env_rules.lp", facts)[0]
        print(f"=== Step {start} -> {end}: New environment inferred as '{inferred_env}' (REAL_ENV = {env_name}) ===")

        agent.load_q_table(inferred_env)

        while step < end:
            state = human.get_state()
            action, explored_state = agent.choose_best_action(state)
            
            reaction = human.get_scaled_reaction(action)
            feedback = environment.get_feedback(action)
            
            human.step(action, feedback)
            
            for emotion in results['history']:
                results['history'][emotion].append(human.emotions[emotion])

            results['actions'].append(action)
            results['explored_states'].append(explored_state)
            
            step += 1
            
    plot_results(results, intervals=widths, envs=envs)

    return results

if __name__ == "__main__":
    ENV_OBS = load_env_obs()
    train(num_episodes=1000,
            max_steps_per_episode=500, 
            epsilon=0.3,
            decay_rate=0.995,
            learning_rate=0.1,
            discount_factor=0.3
            )
    results = simulate_dynamic_session()