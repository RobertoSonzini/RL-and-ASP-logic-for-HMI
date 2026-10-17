import random
import matplotlib.pyplot as plt
import numpy as np

from config import *

class Human:
    """  
        Represents the target human in the environment. The human has an emotional state
        influenced by the actions of the agent and feedback from the social environment.

        Attributes
        ----------
        emotions : dict[str, float]
            Dictionary representing the emotional state of the human target.
            Each emotion value is initialized randomly within a continuous range [min_value, max_value].
                Keys include:
                    - 'happiness'
                    - 'sadness'
                    - 'anger'
                    - 'anxiety'

        reactions : dict[str, dict[str, float]]
            Defines how the target human reacts to the agent's possible actions.
            Each action maps to a dictionary of emotional updates (delta values).

        time_step : int
            Counter of how many interaction steps the human has experienced.
    """

    def __init__(self, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value - 1

        def get_random_init_value_emotions():
            return random.uniform(self.min_value, self.max_value)
        
        def get_random_init_value_deltas(positive, negative):
            if positive and not negative:
                return random.uniform(0.1, 1.0)
            elif negative and not positive:
                return random.uniform(-1.0, -0.1)
            else:
                return random.uniform(-0.5, 0.5)
            
        self.emotions = {
            'happiness': get_random_init_value_emotions(),
            'sadness': get_random_init_value_emotions(),
            'anger': get_random_init_value_emotions(),
            'anxiety': get_random_init_value_emotions()
        }

        self.reactions = {
            "joke": 
                    {"happiness": get_random_init_value_deltas(positive=True, negative=False),
                     "sadness": get_random_init_value_deltas(positive=False, negative=True),
                     "anger": get_random_init_value_deltas(positive=False, negative=True),
                     "anxiety": get_random_init_value_deltas(positive=False, negative=True)},
            "hug": 
                    {"happiness": get_random_init_value_deltas(positive=True, negative=False),
                     "sadness": get_random_init_value_deltas(positive=False, negative=True),
                     "anger": get_random_init_value_deltas(positive=False, negative=True),
                     "anxiety": get_random_init_value_deltas(positive=False, negative=False)},
            "raise_tone": 
                    {"happiness": get_random_init_value_deltas(positive=False, negative=True),
                     "sadness": get_random_init_value_deltas(positive=True, negative=False),
                     "anger": get_random_init_value_deltas(positive=True, negative=False),
                     "anxiety": get_random_init_value_deltas(positive=True, negative=False)},
            "calm_tone": 
                    {"happiness": get_random_init_value_deltas(positive=False, negative=False),
                     "sadness": get_random_init_value_deltas(positive=False, negative=False),
                     "anger": get_random_init_value_deltas(positive=False, negative=True),
                     "anxiety": get_random_init_value_deltas(positive=False, negative=True)},
            "do nothing": 
                    {"happiness": get_random_init_value_deltas(positive=False, negative=False),
                     "sadness": get_random_init_value_deltas(positive=False, negative=False),
                     "anger": get_random_init_value_deltas(positive=False, negative=False),
                     "anxiety": get_random_init_value_deltas(positive=False, negative=False)}
        }

        self.time_step = 0

    def get_state(self):
        return self.emotions.copy()
    
    def apply_action(self, action):
        if action not in self.reactions.keys():
            raise ValueError(f"Action '{action}' not defined in reactions.")

        deltas = self.get_scaled_reactions()[action]
        for emotion, delta in deltas.items():
            self.emotions[emotion] += delta
            self.emotions[emotion] = min(max(self.emotions[emotion], self.min_value), self.max_value)

        return deltas
    
    def apply_social_feedback(self, feedback):
        for emotion, delta in feedback.items():
            self.emotions[emotion] += delta
            self.emotions[emotion] = min(max(self.emotions[emotion], self.min_value), self.max_value)

        return feedback

    def get_reactions(self):
        return self.reactions
    
    def get_scaled_reactions(self):
        scaled_reactions = {}

        emotions_array = np.array([self.emotions[emotion] for emotion in self.emotions])

        for action, deltas in self.reactions.items():
            deltas_array = np.array([deltas[emotion] for emotion in self.emotions])

            scaled_array = deltas_array * np.exp(-emotions_array / 2)

            scaled_deltas = {emotion: scaled_array[i] for i, emotion in enumerate(self.emotions)}
            scaled_reactions[action] = scaled_deltas

        return scaled_reactions
    
    def get_reaction(self, action):
        return self.reactions[action]
    
    def get_scaled_reaction(self, action):
        return self.get_scaled_reactions()[action]
    
    def step(self, action, feedback):
        '''
            A single function to apply all the steps in the internal state.
            Returns the final delta as sum of the 2 contributions (state transition function's value). 
        '''
        delta_human = self.apply_action(action)
        delta_env = self.apply_social_feedback(feedback)

        self.time_step += 1

        delta_total = {k: delta_human.get(k, 0) + delta_env.get(k, 0)
                   for k in set(delta_human) | set(delta_env)}

        return delta_total
    
    def plot_human_state(self):
        emotions = list(self.emotions.keys())
        values = [self.emotions[emotion] for emotion in emotions]

        plt.figure(figsize=(8, 5))
        bars = plt.bar(emotions, values, color=['#FFD700', '#1E90FF', '#FF4500', '#8A2BE2'])
        
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.05, f"{yval:.2f}", 
                    ha='center', va='bottom', fontsize=10)
        
        plt.ylim(self.min_value, self.max_value)
        plt.ylabel("Emotion value")
        plt.title(f"Human emotional state at time step {self.time_step}")
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.show()

    def plot_reactions(self, resolution=100):
        actions = list(self.reactions.keys())
        emotions = list(self.emotions.keys())
        emotion_range = np.linspace(self.min_value, self.max_value, resolution)
        colors = ["#2b7dbf", "#E77F35", "#D6B12B", "#7836B6"]

        fig, axes = plt.subplots(len(actions), 1, figsize=(10, 3*len(actions)), sharex=True)

        for i, action in enumerate(actions):
            ax = axes[i]
            for j, emotion in enumerate(emotions):
                base_delta = self.reactions[action][emotion]
                scaled_values = base_delta * np.exp(-emotion_range / 2)
                ax.plot(emotion_range, scaled_values, label=emotion, color=colors[j], linewidth=2)
            
            ax.set_ylabel("Scaled delta")
            ax.set_title(f"Action: {action}")
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend(loc='upper right')

        axes[-1].set_xlabel("Emotion value")
        plt.tight_layout()
        plt.show()

    def clone(self):
        cloned = Human(self.min_value, self.max_value + 1)
        cloned.emotions = self.emotions.copy()
        cloned.reactions = {action: deltas.copy() for action, deltas in self.reactions.items()}
        cloned.time_step = self.time_step
        return cloned

    def __repr__(self):
        emotions_str = "\n    ".join(
            f"{emotion}: {value:.2f}" for emotion, value in self.emotions.items()
        )

        reactions_lines = []
        for action, effects in self.reactions.items():
            effect_str = "\n      ".join(
                f"{emotion}: {delta:+.2f}"
                for emotion, delta in effects.items()
            )
            reactions_lines.append(f"  {action}:\n      {effect_str}")
        reactions_str = "\n".join(reactions_lines)

        return (
            "HumanTarget(\n"
            f"  time_step = {self.time_step},\n"
            f"  emotions = {{\n    {emotions_str}\n  }},\n"
            f"  reactions = {{\n{reactions_str}\n  }}\n"
            ")"
        )
    
if __name__ == '__main__':
    human = Human(min_value=settings.get_min_value(), max_value=settings.get_max_value())
    print(human.get_reaction("joke"))
    print(human)