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

def run_training_average_q(num_episodes=200, max_steps_per_episode=50, epsilon = 0.7, decay_rate = 0.9995, learning_rate = 0.2, discount_factor = 0.7, context_type = 'neutral', plot_rewards = True, plot_epsilon_trend = True, plot_convergence_trend = True):
    all_q_tables = []
    reward_history = []
    epsilon_histories = []
    convergences = []

    for episode in range(num_episodes):
        human = Human(settings.get_min_value(), settings.get_max_value())
        environment = SocialEnvironment(context_type = context_type)

        episode_agent = Agent(settings.get_min_value(), settings.get_max_value(), settings.get_step_size())

        episode_agent.learning_mode(
            epsilon=epsilon,
            decay_rate=decay_rate,
            learning_rate=learning_rate,
            discount_factor=discount_factor
        )

        total_reward = 0

        for step in range(max_steps_per_episode):
            human_state = human.get_state()
            action = episode_agent.choose_action(human_state)
            
            reaction = human.get_scaled_reaction(action)
            feedback = environment.get_feedback(action)
            
            human.step(action, feedback)
            
            next_human_state = human.get_state()
            
            reward = episode_agent.step(action, human_state, next_human_state, reaction, feedback)
            total_reward += reward
        
        all_q_tables.append(episode_agent.Q)
        reward_history.append(total_reward)
        epsilon_histories.append(episode_agent.get_epsilon_history())
        convergences.append(episode_agent.get_convergence())

    final_Q = {}
    states = all_q_tables[0].keys()
    actions = list(all_q_tables[0][next(iter(states))].keys())

    for state in states:
        final_Q[state] = {}
        for action in actions:
            final_Q[state][action] = sum(q_table[state][action] for q_table in all_q_tables) / num_episodes

    final_agent = Agent(settings.get_min_value(), settings.get_max_value(), settings.get_step_size())
    final_agent.Q = final_Q

    if plot_rewards:
        plot_reward_history(reward_history)

    if plot_epsilon_trend:
        plot_epsilon_histories(epsilon_histories)

    if plot_convergence_trend:
        plot_convergences(convergences)

    return final_agent

def train(num_episodes = 100, max_steps_per_episode=50, epsilon = 0.7, decay_rate = 0.9995, learning_rate = 0.2, discount_factor = 0.7):
    environments = ['neutral', 'formal', 'friendly', 'tense', 'hostile']
    all_q_tables = {}

    for env_type in environments:
        print(f"Training agent for environment: {env_type}")
        agent = run_training_average_q(num_episodes=num_episodes, max_steps_per_episode=max_steps_per_episode, context_type=env_type, epsilon = epsilon, decay_rate = decay_rate, learning_rate = learning_rate, discount_factor = discount_factor, plot_rewards = False, plot_epsilon_trend = False, plot_convergence_trend = False)

        q_table_serializable = {str(state): actions for state, actions in agent.Q.items()}

        all_q_tables[env_type] = q_table_serializable

    with open("q_tables.json", "w") as f:
        json.dump(all_q_tables, f, indent=2)

    print("All Q-tables saved in q_tables.json")

def train_baseline(num_episodes = 100, max_steps_per_episode=50, epsilon = 0.7, decay_rate = 0.9995, learning_rate = 0.2, discount_factor = 0.7, env_type = None):
    environments = ['neutral', 'formal', 'friendly', 'tense', 'hostile']

    env_type = random.choice(environments) if env_type is None else env_type

    print(f"Training baseline agent for environment: {env_type}")
    agent = run_training_average_q(num_episodes=num_episodes, max_steps_per_episode=max_steps_per_episode, context_type=env_type, epsilon = epsilon, decay_rate = decay_rate, learning_rate = learning_rate, discount_factor = discount_factor, plot_rewards = False, plot_epsilon_trend = False, plot_convergence_trend = False)

    q_table_serializable = {str(state): actions for state, actions in agent.Q.items()}

    with open("baseline_q_table.json", "w") as f:
        json.dump(q_table_serializable, f, indent=2)

    print("Baseline q-table saved in baseline_q_table.json")