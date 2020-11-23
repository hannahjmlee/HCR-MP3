import numpy as np
import random

class DecisionRule():
    def __init__(self):
        self.agent_prob = 0.1

    def get_action(self, state, user_action):
        if user_action != -1:
            return user_action
        return 0

