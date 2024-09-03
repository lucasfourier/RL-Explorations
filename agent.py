import numpy as np

class Agent:
    """
    """
    def __init__(self, num_actions = 10, epsilon = 0.1, steps = 2000):
        """Constructor of the Agent class.

        Args:
            num_actions (int, optional): Number of decisions to make. Defaults to 10.
            epsilon (float, optional): Adjustable parameter for exploration/exploitation. Defaults to 0.1.
            steps (int, optional): Number of plays. Defaults to 2000.
        """
        self.q_estimates = np.zeros(num_actions)
        self.action_count = np.zeros(num_actions)
        self.cumulative_rewards_received = np.zeros(num_actions)
        self.epsilon = epsilon

    def get_Q_estimates(self) -> np.ndarray:
        """Returns Q np.ndarray

        Returns:
            np.ndarray: _description_
        """
        return self.q_estimates
    
    def get_action_count(self) -> np.ndarray:
        """Returns np.ndarray containing how 
        many times each action was taken.

        Returns:
            np.ndarray: #Times each action was taken.
        """
        return self.action_count
    
    def get_cumulative_rewards(self) -> np.ndarray:
        """Returns the cumulative reward over time
        provided by each action.

        Returns:
            np.ndarray: _description_
        """
        return self.cumulative_rewards_received
    
    def compute_index_candidate_action(self, Q_estimates) -> np.ndarray:
        """Computes the INDEX such that Q() is maximal.

        Args:
            Q_estimates (np.ndarray): Approximations for q.

        Returns:
            np.ndarray: index such that Q() is maximal.
        """
        return np.argmax(Q_estimates)
        
    def compute_argmax_actions(self, Q_estimates: np.ndarray) -> np.ndarray:
        """Computes actions where Q is maximal.

        Args:
            Q_estimates (np.ndarray): Approximations for q.

        Returns:
            np.ndarray: np.ndarray containing actions such that Q is maximal.
        """
        return np.where(Q_estimates == Q_estimates[self.compute_index_candidate_action(Q_estimates)])[0]