import numpy as np

class Agent:
    """
    """
    def __init__(self, num_actions = 10, epsilon = 0.1, steps = 2000):
        """_summary_

        Args:
            num_estimates (int, optional): _description_. Defaults to 10.
        """
        self.q_estimates = np.zeros(num_actions)
        self.action_count = np.zeros(num_actions)
        self.cumulative_rewards_received = np.zeros(num_actions)
        self.epsilon = epsilon

    def get_Q_estimates(self):
        return self.q_estimates
    
    def get_action_count(self):
        return self.action_count
    
    def get_cumulative_rewards(self):
        return self.cumulative_rewards_received
    
    def compute_index_candidate_action(self, Q_estimates):
        return np.argmax(Q_estimates)
        
    def compute_argmax_actions(self, Q_estimates):
        return np.where(Q_estimates == Q_estimates[self.compute_index_candidate_action(Q_estimates)])[0]