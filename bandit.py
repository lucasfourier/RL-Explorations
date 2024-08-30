import numpy as np

class Bandit:
    """_summary_
    """
    def __init__(self, num_actions=10):
        """_summary_

        Args:
            num_actions (int, optional): _description_. Defaults to 10.
        """
        self.num_actions = num_actions
        self.true_value_actions = np.zeros(num_actions)
        self.rewards = np.zeros(num_actions)

    def fill_true_action_values(self, mean=0, standard_deviation=1) -> None:
        """The true value q*(a) of each of the actions was selected
        according to a normal distribution with mean zero and unit
        variance. (pg 28)

        Args:
            mean (int, optional): _description_. Defaults to 0.
            standard_deviation (int, optional): _description_. Defaults to 1.
        """
        for a in range(self.num_actions):
            self.true_value_actions[a] = np.random.normal(mean, standard_deviation)

    def fill_reward_values(self, true_value_actions) -> None:
        """Actual rewards were selected according to a mean
        q*(a), for each action 'a', and unit variance. (pg 28)

        Args:
            true_value_actions (_type_): _description_
        """
        for a in range(self.num_actions):
            #print(f"Reward {a} based off true value q*({a}) = {true_value_actions[a]}")
            self.rewards[a] = np.random.normal(true_value_actions[a], 1)

    def get_action_values(self) -> np.ndarray:
        """Returns the true action values q*(a)
        which were defined by 'fill_true_action_values'.

        Returns:
            _type_: _description_
        """
        return self.true_value_actions

    def get_reward_values(self) -> np.ndarray:
        """Returns the rewards which were defined
        by method 'fill_reward_values'

        Returns:
            _type_: _description_
        """
        return self.rewards