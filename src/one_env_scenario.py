import matplotlib.pyplot as plt
import copy
import json
import random

from human import Human
from environment import SocialEnvironment
from agent import Agent

from config import *
from plots import *

def run_training_average_q(num_episodes=200, max_steps_per_episode=50, context_type = 'neutral', plot_rewards = True, plot_epsilon_trend = True, plot_convergence_trend = True):
    all_q_tables = []
    reward_history = []
    epsilon_histories = []
    convergences = []

    for episode in range(num_episodes):
        human = Human(settings.get_min_value(), settings.get_max_value())
        environment = SocialEnvironment(context_type = context_type)

        episode_agent = Agent(settings.get_min_value(), settings.get_max_value(), settings.get_step_size())

        episode_agent.learning_mode(
            epsilon=0.3,
            decay_rate=0.995,
            learning_rate=0.1,
            discount_factor=0.3
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

def simulate_until_happy(agent, context_type = 'neutral', environment=None, max_steps=100, plots = True):
    results = {'history': {'happiness': [], 'sadness': [], 'anger': [], 'anxiety': []}, 
               'actions': [],
               'explored_states': []}
    
    human = Human(settings.get_min_value(), settings.get_max_value())
    while human.emotions['happiness'] <= settings.get_max_value() - 1 and human.emotions['happiness'] > settings.get_max_value() - 2:
        human = Human(settings.get_min_value(), settings.get_max_value())

    environment = environment or SocialEnvironment(context_type)

    if plots:
        human.plot_human_state()
        human.plot_reactions()
        plot_total_reactions(human, environment.get_feedbacks())
    
    step = 0
    while human.emotions['happiness'] < settings.get_max_value() - 1 and step < max_steps:
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
    
    return results

def test(num_episodes=50, max_steps_per_episode=50, path = 'deviations.json'):
    with open(path, "r") as f:
        deviations = json.load(f)

    context_types = list(deviations.keys()) 

    for context_type in context_types:
        print(context_type)
        agent = run_training_average_q(num_episodes=num_episodes, max_steps_per_episode=max_steps_per_episode, context_type=context_type)

        for i in range(5):
            results = simulate_until_happy(agent, context_type=context_type)
            plot_results(results)

def single_test(num_episodes=800, max_steps_per_episode=500, path = 'deviations.json'):
    with open(path, "r") as f:
        deviations = json.load(f)

    context_types = list(deviations.keys()) 

    context_type = random.choice(context_types)
    print(f"Selected context type: {context_type}")

    agent = run_training_average_q(num_episodes=num_episodes, 
                                max_steps_per_episode=max_steps_per_episode, 
                                context_type=context_type)
    
    plot_q_table_analysis(agent)

    for i in range(5):
        results = simulate_until_happy(agent, context_type=context_type)
        plot_results(results)

def test_statistical(num_episodes=50, max_steps_per_episode=50, path='deviations.json',
                     num_simulations=20, happiness_pct=0.7):
    with open(path, "r") as f:
        deviations = json.load(f)

    context_types = list(deviations.keys())
    max_val = settings.get_max_value()
    happiness_threshold = happiness_pct * (max_val - 1)

    stats_by_context = {}

    for context_type in context_types:
        print(f"\n=== Context: {context_type} ===")
        agent = run_training_average_q(num_episodes=num_episodes,
                                       max_steps_per_episode=max_steps_per_episode,
                                       context_type=context_type, plot_rewards= False)

        results = []
        for sim in range(num_simulations):
            results = simulate_until_happy(agent, context_type=context_type)
            history = results['history']
            final_state = {emotion: values[-1] for emotion, values in history.items()}
            results.append(final_state)

        all_states = {key: np.array([r[key] for r in results]) for key in results[0]}

        context_stats = {}
        for emotion, values in all_states.items():
            if emotion == 'happiness':
                above_threshold = np.sum(values >= happiness_threshold)
            else:
                above_threshold = None
            context_stats[emotion] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'above_threshold': above_threshold
            }

        stats_by_context[context_type] = context_stats

        print(f"Happiness >= {happiness_pct*100:.0f}% max: {context_stats['happiness']['above_threshold']}/{num_simulations}")
        print(f"Mean/Std: \nHappiness [mean: {context_stats['happiness']['mean']:.2f} - std:{context_stats['happiness']['std']:.2f}]")

    return stats_by_context

if __name__ == '__main__':
    single_test()