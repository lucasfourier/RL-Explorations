import numpy as np

class Bandit:
    """_summary_
    """
    def __init__(self, num_actions=10):
        """Constructor of the class.

        Args:
            num_actions (int, optional): _description_. Defaults to 10.
        """
        self.num_actions = num_actions
        self.true_value_actions = np.zeros(num_actions)
        self.rewards = np.zeros(num_actions)

    def fill_true_action_values_uniform(self, low = -1, high = 1) -> None:
        """The true values q*(a) here of each of the actions are
        selected according to an uniform distribution from -1 to 1.

        Args:
            low (int, optional): lower bound of uniform distribution. Defaults to -1.
            high (int, optional): upper bound of uniform distribution. Defaults to 1.
        """
        for a in range(self.num_actions):
            self.true_value_actions[a] = np.random.uniform(low, high)

    def fill_reward_values_uniform(self, true_value_actions: np.ndarray) -> None:
        """Actual rewards were selected according to an uniform
        distribution with low = q*(a) - 0.1 and high = q*(a) + 0.1
        #TODO: CHECK BOUNDS and provide meaning for those bounds.

        Args:
            true_value_actions (np.ndarray): Actual q*, true value of action.
        """

        for a in range(self.num_actions):
            self.rewards[a] = np.random.uniform(true_value_actions[a] - 0.1, true_value_actions[a] + 0.1)

    def fill_true_action_values(self, mean=0, standard_deviation=1) -> None:
        """The true value q*(a) of each of the actions was selected
        according to a normal distribution with mean zero and unit
        variance. (pg 28)

        Args:
            mean (int, optional): Defaults to 0.
            standard_deviation (int, optional): Defaults to 1.
        """
        for a in range(self.num_actions):
            self.true_value_actions[a] = np.random.normal(mean, standard_deviation)

    def fill_reward_values(self, true_value_actions: np.ndarray) -> None:
        """Actual rewards were selected according to a mean
        q*(a), for each action 'a', and unit variance. (pg 28)

        Args:
            true_value_actions (np.ndarray): Actual q*, true value of action.
        """

        for a in range(self.num_actions):
            self.rewards[a] = np.random.normal(true_value_actions[a], 1)

    def get_action_values(self) -> np.ndarray:
        """Returns the true action values q*(a)
        which were defined by 'fill_true_action_values'.

        Returns:
            np.ndarray: True action values q* for the bandit.
        """
        return self.true_value_actions

    def get_reward_values(self) -> np.ndarray:
        """Returns the rewards which were defined
        by method 'fill_reward_values'

        Returns:
            np.ndarray: Reward values based off q*.
        """
        return self.rewards