import json
from human import Human

class SocialEnvironment:
    def __init__(self, context_type='neutral', path="deviations.json"):
        self.context_type = context_type

        with open(path, "r") as f:
            self.deviations = json.load(f)

        for context in self.deviations:
            for action in self.deviations[context]:
                action_values = self.deviations[context][action]
                max_abs = max(abs(val) for val in action_values.values())
                if max_abs > 0:
                    for emotion in action_values:
                        action_values[emotion] /= max_abs

                        action_values[emotion] /= 5

        self.time_step = 0

    def get_feedback(self, action):
        '''
            Based on the action of the agent, we compute a feedback in terms of deviations.
        '''
        default = {'happiness': 0,
                    'sadness': 0,
                    'anger': 0,
                    'anxiety': 0}
        
        deviations_per_context = self.deviations.get(self.context_type, {})

        return deviations_per_context.get(action, default)
    
    def get_feedbacks(self):
        default = {'happiness': 0,
                    'sadness': 0,
                    'anger': 0,
                    'anxiety': 0}
        
        deviations_per_context = self.deviations.get(self.context_type, default)

        return deviations_per_context