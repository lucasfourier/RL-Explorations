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
        self.action_candidate_array = np.zeros(num_actions)
        self.epsilon = epsilon
        self.steps = steps
        self.exploring = 0
        self.exploiting = 0
        self.candidate_action = 0
        self.A = 0

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
    
    def get_steps(self) -> int:
        """Returns the amount of steps the agent will take.

        Returns:
            int: number of iterations with the bandit machine.
        """
        return self.steps
    
    def get_exploring_count(self) -> int:
        """Returns the amount of times the agent explored.

        Returns:
            int: Amount of times agent explored.
        """
        return self.exploring
    
    def get_exploit_count(self) -> int:
        """Returns the amount of times the agent exploited.

        Returns:
            int: Amount of times the agent exploited.
        """
        return self.exploiting
    
    def set_zero_exploit_count(self) -> None:
        """
        Sets the exploit index to zero
        """
        self.exploiting = 0

    def set_zero_explore_count(self) -> None:
        """
        Sets the explore index to zero
        """
        self.exploring = 0

    def set_zero_cumulative_rewards(self) -> None:
        """
        Sets the cumulative rewards attribute to zero.
        """
        for i in range(len(self.cumulative_rewards_received)):
            self.cumulative_rewards_received[i] = 0

    def set_zero_action_count(self) -> None:
        """
        Sets the action count to zero.
        """
        for i in range(len(self.action_count)):
            self.action_count[i] = 0
    
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
    
    def action_method(self, method = "epsilon-greedy", epsilon = 0.1, number_of_arms = 10, rewards = np.zeros(10)) -> None:
        """Main driver of the agent. It will act according to the method given by the user. 

        Args:
            method (str, optional): strategy chosen. Defaults to "epsilon-greedy".
            epsilon (float, optional): threshold for exploration/exploitation. Defaults to 0.1.
            number_of_arms (int, optional): Number of actions. Defaults to 10.
            rewards (np.ndarray, optional): rewards at some time t. Defaults to np.zeros(10).

        "epsilon-greedy": This method alternates between choosing the action with the
        highest estimated value and sometimes exploring alternative actions with probability
        epsilon.
        """
        self.epsilon = epsilon
        if method == "epsilon-greedy":
            if np.random.random() < self.epsilon:
                self.A = np.random.choice(number_of_arms)

                self.cumulative_rewards_received[self.A] = self.cumulative_rewards_received[self.A] + rewards[self.A]
                self.action_count[self.A] = self.action_count[self.A] + 1
                self.q_estimates[self.A] = self.q_estimates[self.A] + (1/self.action_count[self.A]) * (rewards[self.A] - self.q_estimates[self.A])
                self.exploring = self.exploring + 1
            else:
                self.candidate_action = np.argmax(self.q_estimates)
                self.action_candidate_array = np.where(self.q_estimates == self.q_estimates[self.candidate_action])[0]
                self.A = np.random.choice(self.action_candidate_array)

                self.cumulative_rewards_received[self.A] = self.cumulative_rewards_received[self.A] + rewards[self.A]
                self.action_count[self.A] = self.action_count[self.A] + 1
                self.q_estimates[self.A] = self.q_estimates[self.A] + (1/self.action_count[self.A]) * (rewards[self.A] - self.q_estimates[self.A])
                self.exploiting = self.exploiting + 1

