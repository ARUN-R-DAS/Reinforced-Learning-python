# CHATGPT Helped me organize the code. I also added visuals using pygame

import numpy as np
import random
import pygame
import time

#------------------------ PARAMETERS ------------------------
RESET_Q_TABLE = True
SAVE_Q_TABLE = False
LOAD_Q_TABLE = False

ALPHA = 0.1   # learning rate
GAMMA = 0.9   # discount factor
EPSILON = 0.2 # exploration rate
GOAL = 5      # number of states
ACTIONS = [0, 1]  # 0=left, 1=right

#------------------------ PYGAME SETUP ------------------------
pygame.init()
screen = pygame.display.set_mode((800, 400))
pygame.display.set_caption("Reinforcement Learning Q-table")
clock = pygame.time.Clock()

#------------------------ ENVIRONMENT ------------------------
def step(state, action):
    """Take an action in the environment and return next_state, reward, done"""
    if action == 0:  # move left
        next_state = max(0, state - 1)
    else:  # move right
        next_state = min(GOAL, state + 1)
    
    reward = 1 if next_state == GOAL else 0
    done = next_state == GOAL
    return next_state, reward, done

#------------------------ Q-LEARNING ------------------------
def choose_action(state, q_table):
    """Choose an action using epsilon-greedy policy"""
    if random.random() < EPSILON:
        return random.choice(ACTIONS)
    else:
        return int(np.argmax(q_table[state]))

def update_q_table(q_table, state, action, reward, next_state):
    """Perform the Q-learning update"""
    old_q = q_table[state, action]
    best_future_q = np.max(q_table[next_state])
    q_table[state, action] = old_q + ALPHA * (reward + GAMMA * best_future_q - old_q)

#------------------------ VISUALIZATION ------------------------
def update_frame(state):
    """Draw the squares"""
    screen.fill((0, 0, 0))
    for i in range(GOAL):
        color = (0, 255, 0) if i == state else (255, 255, 255)
        border = 0 if i == state else 2
        pygame.draw.rect(screen, color, (100 + i*100, 100, 100, 100), border)
    pygame.display.flip()

#------------------------ TRAINING ------------------------
def train_one_episode(q_table):
    """Run a single episode of training and return steps taken"""
    state = 0
    done = False
    steps = 0

    while not done:
        # Keep Pygame responsive
        pygame.event.pump()

        action = choose_action(state, q_table)
        next_state, reward, done = step(state, action)
        update_q_table(q_table, state, action, reward, next_state)

        state = next_state
        steps += 1

        # Optional: print progress
        for i in range(GOAL):
            print('+' if i == state else '-', end='')
        print(f" - Step {steps}")

        # Update Pygame visualization
        update_frame(state)
        pygame.time.delay(100)  # slow down for visualization

    return steps

#------------------------ MAIN ------------------------
def main():
    # Q-table initialization
    if RESET_Q_TABLE:
        q_table = np.zeros((GOAL + 1, len(ACTIONS)))
    elif LOAD_Q_TABLE:
        q_table = np.load("q_table.npy")

    running = True
    episode = 0

    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Run one episode per loop iteration
        episode += 1
        print(f"\nEpisode {episode}")
        steps = train_one_episode(q_table)
        print(f"Episode {episode} finished in {steps} steps")

        clock.tick(60)  # optional FPS limit

    if SAVE_Q_TABLE:
        np.save("q_table.npy", q_table)

    pygame.quit()

#------------------------ ENTRY POINT ------------------------
if __name__ == "__main__":
    main()