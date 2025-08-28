import os
import numpy as np
import matplotlib.pyplot as plt
from snake_env import SnakeGame
from dqn_agent import DQNAgent
import torch

if __name__ == "__main__":
    print("Starting training...")
    if not os.path.exists('models'):
        os.makedirs('models')
    print("Models directory created or exists.")
    try:
        game = SnakeGame(None)
        print("SnakeGame initialized successfully.")
        agent = DQNAgent(state_size=11, action_size=3)
        print("DQNAgent initialized successfully.")
        game.agent = agent
    except Exception as e:
        print(f"Initialization error: {e}")
        exit(1)
    episodes = 1000
    batch_size = 32
    max_score = 0

    for e in range(episodes):
        state = game.reset()
        score = 0
        while True:
            action = agent.act(state)
            reward, done, score, game_count = game.play_step(action)
            next_state = game._get_state()
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            agent.replay(batch_size)
            if done:
                game.scores.append(score)
                print(f"Episode: {e+1}/{episodes}, Game: {game_count}, Score: {score}, Epsilon: {agent.epsilon:.2f}")
                model_filename = os.path.join('models', f"model_episode_{e+1}.pth")
                torch.save(agent.model.state_dict(), model_filename)
                np.save(os.path.join('models', f"scores_episode_{e+1}.npy"), np.array(game.scores))
                if score > max_score:
                    max_score = score
                    torch.save(agent.model.state_dict(), os.path.join('models', "model_best.pth"))
                break
        if e % 50 == 0:
            agent.update_target_model()

    plt.figure(figsize=(10, 6))
    plt.plot(game.scores, label='Raw Scores', color='purple', alpha=0.5)
    if len(game.scores) >= 100:
        window = np.ones(100) / 100
        moving_avg = np.convolve(game.scores, window, mode='valid')
        plt.plot(range(99, len(game.scores)), moving_avg, label='100-Episode Running Average', color='red', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_progress.png')
    plt.show()