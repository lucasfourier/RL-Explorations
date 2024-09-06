import numpy as np
import matplotlib.pyplot as plt

from agent import Agent
from bandit import Bandit

# Creating a bandit with 'n' arms/levers.

number_of_arms = 10
bandit = Bandit(number_of_arms)

# Filling/computing q*(a), for each 'a'. Bandit Intrinsic.
# Remember that: q*(a) = True value of an action.

bandit.fill_true_action_values(0, 1)
q_star = bandit.get_action_values()

# Filling/computing rewards of each action. 

bandit.fill_reward_values(q_star)
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

for i in range(steps):
    # Play!
    # Rewards are updated each time.
    #print(f"Iteration {i}")
    bandit.fill_reward_values(q_star)
    rewards = bandit.get_reward_values()
    
    if np.random.random() < epsilon:
        #print(f"-> epsilon-greedy. Exploring.")
        A = np.random.choice(number_of_arms)
        #print(f"Chose action {A} => reward = {rewards[A]}")
        cumulative_rewards_received[A] = cumulative_rewards_received[A] + rewards[A]
        action_count[A] = action_count[A] + 1
        Q_estimates[A] = Q_estimates[A] + (1/action_count[A]) * (rewards[A] - Q_estimates[A])
        exploring = exploring + 1

    else:
        #print(f"-> Greedy. Exploiting.")
        candidate_action = np.argmax(Q_estimates)
        action_candidate_array = np.where(Q_estimates == Q_estimates[candidate_action])[0]

        # If multiple values with same action value, random choice.
        A = np.random.choice(action_candidate_array)
        #print(f"Chose action {A} => reward = {rewards[A]}")
        cumulative_rewards_received[A] = cumulative_rewards_received[A] + rewards[A]
        action_count[A] = action_count[A] + 1
        Q_estimates[A] = Q_estimates[A] + (1/action_count[A]) * (rewards[A] - Q_estimates[A])
        exploiting = exploiting + 1

print("END")
print(f"Q = {Q_estimates}")
print(f"True values = {q_star}")
print(f"(Average) Cumulative reward = {cumulative_rewards_received/steps}")
print(f"action count = {action_count}")
print(f"Exploited {exploiting} and explored {exploring}")
print(f"Average reward = {np.sum(cumulative_rewards_received)/steps:.3f}")