import torch
from game import SnakeGameAI
from model import Linear_QNet
from agent import Agent


def play():
    """Load the trained model and watch the AI play."""
    agent = Agent()
    # Load saved weights
    model_path = './models/model.pth'
    agent.model.load_state_dict(torch.load(model_path))
    agent.model.eval()

    game = SnakeGameAI()

    while True:
        state = agent.get_state(game)
        state0 = torch.tensor(state, dtype=torch.float)
        prediction = agent.model(state0)
        move = torch.argmax(prediction).item()

        final_move = [0, 0, 0]
        final_move[move] = 1

        reward, done, score = game.play_step(final_move)
        if done:
            print(f'Final Score: {score}')
            game.reset()


if __name__ == '__main__':
    play()