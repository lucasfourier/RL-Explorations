import numpy as np
import matplotlib.pyplot as plt

from agent import Agent
from bandit import Bandit

# Creating a bandit with 'n' arms/levers.

number_of_arms = 10
bandit = Bandit(number_of_arms)

# Filling/computing q*(a), for each 'a'. Bandit Intrinsic.
# Remember that: q*(a) = True value of an action.

#bandit.fill_true_action_values(0, 1)
bandit.fill_true_action_values_uniform(-1, 1)
q_star = bandit.get_action_values()

# Filling/computing rewards of each action. 

#bandit.fill_reward_values(q_star)
bandit.fill_reward_values_uniform(q_star)
rewards = bandit.get_reward_values()

### Prototype
agent = Agent(num_actions=10, epsilon=0.1, steps = 2000)

# ndarrays to keep Q_estimates and action_count.
Q_estimates = agent.get_Q_estimates()
action_count = agent.get_action_count()

# Initializing actions. 
# Here, any action is a candidate.
candidate_action = agent.compute_index_candidate_action(Q_estimates)

# 'action' keeps the indexes where Q_(.) is maximum.
action = agent.compute_argmax_actions(Q_estimates)
#print(f"Actions such that Q is max = {action}")

# Vector to keep cumulative rewards. Updated each step.
cumulative_rewards_received = agent.get_cumulative_rewards()

### Prototype

# Some parameters

epsilon = 0.1
steps = 2000


# Just to count the number of times it exploited/explored.

exploring = agent.get_exploring_count()
exploiting = agent.get_exploit_count()

for i in range(2000):
    bandit.fill_reward_values(q_star)
    rewards = bandit.get_reward_values()

    agent.action_method("epsilon-greedy", epsilon, number_of_arms, rewards)

print("END")
print(f"Q = {agent.get_Q_estimates()}")
print(f"True values = {q_star}")
print(f"True values (mean) = {np.sum(q_star)/10}")
print(f"(Average) Cumulative reward = {agent.get_cumulative_rewards()/2000}")
print(f"action count = {agent.get_action_count()}")
print(f"Exploited {agent.get_exploit_count()} and explored {agent.get_exploring_count()}")
print(f"Average reward = {np.sum(agent.get_cumulative_rewards())/2000:.3f}")



plt.hist(q_star)
plt.show()