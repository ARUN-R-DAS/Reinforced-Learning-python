import numpy as np
import random
import time

#------------- Modify these parameters---------------------
reset_q_table_on_runtime = True
save_q_table = False
load_q_table = False
#____________________________ENVIRONMENT SETUP______________________________
GOAL = 5

# Q-table: 6 states, 2 actions (0=left, 1=right)
if reset_q_table_on_runtime:
    q_table = np.zeros((6,2)) #------------------------------RESET THE Q-TABLE
if load_q_table:
    q_table = np.load("q_table.npy") #-------------------------LOADING SAVED Q-TABLE

alpha = 0.1 # learning rate
gamma = 0.9 # discount factor
epsilon = 0.2 # randomness

#____________________________ENVIRONMENT______________________________

def step(state, action):
    if action == 0: # left
        next_state = max(0, state - 1)
    else:
        next_state = min(GOAL, state + 1)
    
    reward = 1 if next_state == GOAL else 0
    done = next_state == GOAL

    return next_state, reward, done

#____________________________LEARNING LOOP______________________________

for episode in range(10):

    step_count = 0
    state = 0
    done = False

    print(f"\nEpisode {episode + 1}")

    while not done:

        # choose action (exploration vs exploitation)
        if random.random() < epsilon:
            action = random.randint(0, 1)
        else:
            action = np.argmax(q_table[state])
        
        next_state, reward, done = step(state, action)

        # Q-learning update
        old_q = q_table[state, action]
        best_future_q = np.max(q_table[next_state])

        q_table[state, action] = old_q + alpha * (reward + gamma * best_future_q - old_q)

        state = next_state
        step_count += 1
        #_________________________GRAPHICS__________________________
        for i in range(GOAL):
            if i == state:
                print('+',end='')
            else:
                print('-',end='')
        print(f" - {step_count}")

if save_q_table:
    np.save("q_table.npy", q_table)