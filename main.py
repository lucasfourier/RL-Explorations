import numpy as np
import matplotlib.pyplot as plt

from agent import Agent
from bandit import Bandit


# Criar uma máquina com número n de arms.

number_of_arms = 10
bandit = Bandit(number_of_arms)

# Preencher q*(a), para cada 'a'. Intrínsico, não muda.
# Lembrando que: q*(a) = True value of an action.

bandit.fill_true_action_values(0, 1)
q_star = bandit.get_action_values()

# Preencher as rewards de cada ação. 

bandit.fill_reward_values(q_star)
rewards = bandit.get_reward_values()

# Estimativas e vetores de contagem iniciais (inicialização)

Q_estimates = np.zeros(number_of_arms)
action_count = np.zeros(number_of_arms)

# Inicializar as ações

candidate_action = np.argmax(Q_estimates)

# Action guarda os índices nos quais Q_(.) é máximo.

action = np.where(Q_estimates == Q_estimates[candidate_action])[0]
#print(f"Ações onde Q é máximo = {action}")

# Alguns parâmetros da ação

epsilon = 0.1
steps = 2000

# Um último vetor para guardar as recompensas recebidas em cada passo.
cumulative_rewards_received = np.zeros(number_of_arms)

# Just to count the number of times it exploited/explored.

exploring = 0
exploiting = 0

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



