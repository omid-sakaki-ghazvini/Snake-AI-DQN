# 🐍 Snake AI with Deep Q-Network (DQN)

An implementation of a Snake game trained with Deep Reinforcement Learning 
(DQN) using PyTorch and Pygame. The agent learns to play Snake from scratch 
in a few hundred episodes.

![Snake AI Demo](training_progress_final.png)

## 🎯 Features

- **Deep Q-Network (DQN)** implementation from scratch with PyTorch
- **Experience Replay** for stable training
- **Live training visualization** with real-time matplotlib plot in a separate thread
- **Two modes**: Watch the AI play, or play manually yourself
- **Fast training**: optimized for CPU-only machines (no GPU required)
- **Headless mode**: 10-20x faster training without rendering