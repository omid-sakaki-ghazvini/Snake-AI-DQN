import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Agent:
    """DQN Agent for playing Snake."""

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0      # Randomness for exploration
        self.gamma = 0.9      # Discount factor
        self.memory = deque(maxlen=MAX_MEMORY)  # Experience replay buffer
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        """Delegate state extraction to the game environment."""
        return game.get_state()

    def remember(self, state, action, reward, next_state, done):
        """Store transition in replay memory."""
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        """Sample a batch and train on it."""
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        """Train on a single transition (online learning)."""
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        """
        Epsilon-greedy action selection.
        Epsilon decays as the agent plays more games.
        """
        # Linear decay: more games -> less randomness
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]

        if random.randint(0, 200) < self.epsilon:
            # Explore: random action
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            # Exploit: use model prediction
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move