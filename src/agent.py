import random
import json
import numpy as np
import math

from human import Human
from environment import SocialEnvironment

class Agent():
    def __init__(self, min_val, max_val, step_size):
        self.min_val = min_val
        self.max_val = max_val
        self.step_size = step_size
        self.time_step = 0
        
        self.states = [(h, s, a, x) 
                        for h in np.arange(self.min_val, self.max_val, self.step_size) 
                        for s in np.arange(self.min_val, self.max_val, self.step_size)
                        for a in np.arange(self.min_val, self.max_val, self.step_size) 
                        for x in np.arange(self.min_val, self.max_val, self.step_size)]
        self.actions = ['joke', 'hug', 'raise_tone', 'calm_tone', 'do nothing']

        self.Q = {state: {action: 0.0 for action in self.actions} for state in self.states}

        self.epsilon_history = []

        self.convergence = []

    def get_epsilon_history(self):
        return self.epsilon_history

    def learning_mode(self, epsilon, decay_rate, learning_rate, discount_factor):
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.lr = learning_rate
        self.discount_factor = discount_factor

    def discretize_state(self, human_state):
        discrete_state = []

        step = self.step_size
        if self.step_size < 1:
            step = 1/self.step_size

        num_bins = math.ceil((self.max_val - self.min_val) / step)

        for emotion in ['happiness', 'sadness', 'anger', 'anxiety']:
            value = human_state[emotion]
            value = max(self.min_val, min(self.max_val, value))
            bin_index = int((value - self.min_val) / step)
            bin_index = min(bin_index, num_bins - 1)
            discrete_state.append(bin_index)

        return tuple(discrete_state)
    
    def choose_action(self, state):
        discretized_state = self.discretize_state(state)

        if random.random() < self.epsilon:
            action = random.choice(self.actions)
        else:
            q_values = self.Q[discretized_state]
            max_q = max(q_values.values())
            best_actions = [a for a, q in q_values.items() if q == max_q]
            action = random.choice(best_actions)
        return action
    
    def choose_best_action(self, state):
        discretized_state = self.discretize_state(state)

        explored_state = True

        if discretized_state not in self.Q:
            self.Q[discretized_state] = {action: 0 for action in self.actions}
        
        if self.Q[discretized_state] == {action: 0 for action in self.actions}:
            explored_state = False

        q_values = self.Q[discretized_state]
        max_q = max(q_values.values())
        best_actions = [a for a, q in q_values.items() if q == max_q]
        action = random.choice(best_actions)
        return action, explored_state
    
    def get_reward(self, delta_human, env_feedback, weights_human=None, weights_env=None, debug=False):
        default_weights_human = {
            'happiness': 100.0,
            'sadness': -10.0,
            'anger': -10.0,
            'anxiety': -10.0
        }

        default_weights_env = {
            'happiness': 50.0,
            'sadness': -5.0,
            'anger': -5.0,
            'anxiety': -5.0
        }

        weights_human = weights_human or default_weights_human
        weights_env = weights_env or default_weights_env

        reward_human = 0.0
        for emotion in delta_human:
            contrib = delta_human[emotion] * weights_human.get(emotion, 1.0)
            reward_human += contrib
            if debug:
                print(f"[Human] {emotion}: delta={delta_human[emotion]:+.2f}, weight={weights_human.get(emotion,0.0)}, contrib={contrib:+.2f}")

        reward_env = 0.0
        for key in env_feedback:
            contrib = env_feedback[key] * weights_env.get(key, np.sign(env_feedback[key]) * 1.0)
            reward_env += contrib
            if debug:
                print(f"[Env] {key}: feedback={env_feedback[key]:+.2f}, weight={weights_env.get(key,1.0)}, contrib={contrib:+.2f}")

        reward = reward_human + reward_env

        if debug:
            print(f"Total reward = reward_human({reward_human:+.2f}) + reward_env({reward_env:+.2f}) = {reward:+.2f}\n")

        return reward

    def update_q_table(self, state, action, reward, next_state):
        max_next_q = max(self.Q[next_state].values())

        old_q = self.Q[state][action]

        self.Q[state][action] = old_q + self.lr * (reward + self.discount_factor * max_next_q - old_q)

        new_q = self.Q[state][action]

        self.convergence.append(abs(new_q - old_q))

    def step(self, action, current_state, next_state, reaction, feedback):
        state = self.discretize_state(current_state)
        next_state_disc = self.discretize_state(next_state)
        
        reward = self.get_reward(reaction, feedback)

        self.update_q_table(state, action, reward, next_state_disc)
        
        self.time_step += 1
        self.epsilon *= self.decay_rate
        self.epsilon = max(self.epsilon, 0.05)

        self.epsilon_history.append(self.epsilon)

        return reward
    
    def load_q_table(self, inferred_env):
        with open("q_tables.json", "r") as f:
            q_tables = json.load(f)

        Q_str = q_tables[inferred_env]
        
        self.Q = {state: {action: 0.0 for action in self.actions} for state in self.states}
        for state_str, actions in Q_str.items():
            state_tuple = eval(state_str)
            self.Q[state_tuple] = actions

    def load_baseline_q_table(self):
        with open("baseline_q_table.json", "r") as f:
            q_table = json.load(f)
        
        self.Q = {state: {action: 0.0 for action in self.actions} for state in self.states}
        for state_str, actions in q_table.items():
            state_tuple = eval(state_str)
            self.Q[state_tuple] = actions

    def get_epsilon_history(self):
        return self.epsilon_history
    
    def analyze_actual_q_table(self):
        results = {}

        total_states = len(self.Q)
        visited_states = 0

        for state, actions in self.Q.items():
            if any(q != 0.0 for q in actions.values()):
                visited_states += 1

        coverage_ratio = visited_states / total_states

        results["total_states"] = total_states
        results["visited_states"] = visited_states
        results["state_coverage_ratio"] = coverage_ratio

        return results
    
    def get_convergence(self):
        return self.convergence

def trial():
    agent = Agent(0.3, 0.1, 0.1, 0.2)
    human = Human()
    env = SocialEnvironment()
    human_state = human.get_state()
    action_choosen = agent.choose_action(human_state)
    reaction = human.get_reatcion(action_choosen)
    feedback = env.get_feedback(action_choosen)
    print(action_choosen)
    print(reaction)
    print(feedback)
    print(agent.get_reward(reaction, feedback, debug=True))

if __name__ == '__main__':
    trial()