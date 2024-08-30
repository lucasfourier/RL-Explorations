import numpy as np

class Agent:
    """
    """
    def __init__(self, num_estimates = 10):
        """_summary_

        Args:
            num_estimates (int, optional): _description_. Defaults to 10.
        """
        self.estimates = np.zeros(num_estimates)
        self.estimate_action_values = np.zeros(num_estimates)
        self.action_counts = np.zeros(num_estimates)

    def naive_q_estimate(self, rewards) -> np.ndarray:
        """_summary_

        Args:
            rewards (_type_): _description_

        Returns:
            np.ndarray: _description_
        """
        for a in range(len(rewards)):

            self.action_counts[a] += 1
            n = self.action_counts[a]
            self.estimates[a] += (1 / n) * (rewards[a] - self.estimates[a])

        return self.estimates

