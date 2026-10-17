import json
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import numpy as np

class Settings:
    """
    Configuration handler for general project parameters and agent parameters.

    Distinguishes:
    - General parameters (project-wide)
    - Agent parameters (specific to agent behavior)
    Each can have default values and search space for optimization.
    """

    def __init__(self, path: str = "settings.json") -> None:
        with open(path, "r", encoding="utf-8") as f:
            data: Dict[str, Any] = json.load(f)

        general = data.get("general", {})
        self.general_defaults = general.get("defaults", {})

        self.max_value = self.general_defaults.get("max_value", 0)
        self.min_value = self.general_defaults.get("min_value", 0)
        self.step_size = self.general_defaults.get("step_size", 1)
        self.max_window_width = self.general_defaults.get("max_window_width", 10)
        self.min_window_width = self.general_defaults.get("min_window_width", 0)

        agent_params = data.get("agent", {})
        self.agent_defaults = agent_params.get("defaults", {})
        self.agent_search_space = agent_params.get("search_space", {})

        self.default_epsilon = self.agent_defaults.get("epsilon", 0.7)
        self.default_decay_rate = self.agent_defaults.get("decay_rate", 0.9995)
        self.default_learning_rate = self.agent_defaults.get("learning_rate", 0.2)
        self.default_discount_factor = self.agent_defaults.get("discount_factor", 0.7)

    def get_min_value(self):
        return self.min_value

    def get_max_value(self):
        return self.max_value

    def get_step_size(self):
        return self.step_size

    def get_general_params(self):
        """Return dictionary of general default parameters."""
        return self.general_defaults

    def get_agent_params(self):
        """Return dictionary of agent default parameters."""
        return self.agent_defaults

    def get_agent_search_space(self):
        """Return dictionary of agent parameters search space."""
        return self.agent_search_space
    
    def get_max_window_width(self):
        return self.max_window_width
    
    def get_min_window_width(self):
        return self.min_window_width

    def __repr__(self) -> str:
        general_str = ", ".join(f"{k}={v}" for k, v in self.general_defaults.items())
        agent_str = ", ".join(f"{k}={v}" for k, v in self.agent_defaults.items())
        search_str = ", ".join(f"{k}={v}" for k, v in self.agent_search_space.items())
        return (f"Settings(\n"
                f"  General Defaults: {general_str},\n"
                f"  Agent Defaults: {agent_str},\n"
                f"  Agent Search Space: {search_str}\n)")

settings = Settings()

if __name__ == "__main__":
    settings = Settings()
    print(settings)