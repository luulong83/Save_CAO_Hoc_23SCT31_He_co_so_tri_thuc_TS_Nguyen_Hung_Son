from snake_env import SnakeGame
from dqn_agent import DQNAgent
import torch
import os

if __name__ == "__main__":
    game = SnakeGame()
    agent = DQNAgent(state_size=11, action_size=3)
    try:
        agent.model.load_state_dict(torch.load(os.path.join('models', "model_best.pth")))
    except FileNotFoundError:
        print("No trained model found. Please train the model first or load a saved file.")
    agent.epsilon = 0.0
    state = game.reset()
    while True:
        action = agent.act(state)
        reward, done, score, game_count = game.play_step(action)
        state = game._get_state()
        if done:
            print(f"Game Over! Game: {game_count}, Score: {score}")
            break
    pygame.quit()