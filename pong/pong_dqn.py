import random
import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import os
import logging
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from heapq import nlargest

# Thiết lập logging
logging.basicConfig(filename='training.log', level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Khởi tạo Pygame
pygame.init()

# Cài đặt trò chơi
WINDOW_WIDTH = 600
WINDOW_HEIGHT = 400
PADDLE_WIDTH = 10
PADDLE_HEIGHT = 100
BALL_SIZE = 10
PADDLE_SPEED = 5
PADDLE_SPEED_OPPONENT = 5
BALL_SPEED = 4
BLOCK_SIZE = 20

# Màu sắc
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
AI_COLOR = (0, 255, 0)
OPPONENT_COLOR = (255, 0, 0)
BUTTON_COLOR = (0, 100, 0)
BUTTON_HOVER_COLOR = (0, 150, 0)

# Lớp mô hình DQN
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Lớp Agent
class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=5000)
        self.priority = deque(maxlen=5000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.997
        self.learning_rate = 0.0002
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        self.update_target_every = 1000
        self.step_count = 0
        self.alpha = 0.6

    def remember(self, state, action, reward, next_state, done, error=0):
        priority = (abs(error) + 0.01) ** self.alpha
        self.memory.append((state, action, reward, next_state, done))
        self.priority.append(priority)

    def act(self, state):
        logging.debug(f"Input state shape: {state.shape}")
        if random.random() <= self.epsilon:
            action = random.randrange(self.action_size)
            logging.info(f"Random action chosen: {action}")
            return action
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
        logging.debug(f"Q-values: {q_values.cpu().numpy()}")
        action = torch.argmax(q_values, dim=1).item()
        logging.info(f"Model action chosen: {action}")
        return action

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        priorities = np.array(self.priority)
        probabilities = priorities / priorities.sum()
        indices = np.random.choice(len(self.memory), batch_size, p=probabilities)
        minibatch = [self.memory[i] for i in indices]

        states = torch.FloatTensor(np.array([t[0] for t in minibatch])).to(self.device)
        actions = torch.LongTensor([t[1] for t in minibatch]).to(self.device)
        rewards = torch.FloatTensor([t[2] for t in minibatch]).to(self.device)
        next_states = torch.FloatTensor(np.array([t[3] for t in minibatch])).to(self.device)
        dones = torch.FloatTensor([t[4] for t in minibatch]).to(self.device)

        with torch.no_grad():
            next_actions = torch.argmax(self.model(next_states), dim=1)
            targets = rewards + (1 - dones) * self.gamma * self.target_model(next_states)[range(batch_size), next_actions]
        targets_full = self.model(states)
        targets_full[range(batch_size), actions] = targets

        self.optimizer.zero_grad()
        loss = self.criterion(self.model(states), targets_full)
        loss.backward()
        self.optimizer.step()

        new_priorities = (abs(loss.item()) + 0.01) ** self.alpha
        for i in range(batch_size):
            self.priority[indices[i]] = new_priorities

        self.step_count += 1
        if self.step_count % self.update_target_every == 0:
            self.target_model.load_state_dict(self.model.state_dict())
            logging.info("Target model updated")

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            logging.info(f"Epsilon updated: {self.epsilon}")

# Hàm chọn file mô hình
def select_model_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Chọn file mô hình",
        initialdir="models",
        filetypes=[("Model files", "*.pth"), ("All files", "*.*")]
    )
    root.destroy()
    return file_path

# Lớp trò chơi Pong
class PongGame:
    def __init__(self, agent):
        self.display = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption('Pong Game with DQN')
        self.font = pygame.font.SysFont('arial', 20)
        self.button_font = pygame.font.SysFont('arial', 15)
        self.agent = agent
        self.game_count = 0
        self.scores = []
        self.paused = False
        self.showing_progress = False
        self.intelligence_level = 1
        self.previous_y_distance = 0
        self.previous_ai_y = WINDOW_HEIGHT // 2
        self.reset()

    def reset(self):
        self.game_count += 1
        self.ai_paddle = pygame.Rect(50, WINDOW_HEIGHT // 2 - PADDLE_HEIGHT // 2, PADDLE_WIDTH, PADDLE_HEIGHT)
        self.simple_paddle = pygame.Rect(WINDOW_WIDTH - 50 - PADDLE_WIDTH, WINDOW_HEIGHT // 2 - PADDLE_HEIGHT // 2, PADDLE_WIDTH, PADDLE_HEIGHT)
        self.ball = pygame.Rect(WINDOW_WIDTH // 2 - BALL_SIZE // 2, WINDOW_HEIGHT // 2 - BALL_SIZE // 2, BALL_SIZE, BALL_SIZE)
        self.ball_speed = [BALL_SPEED * random.choice((1, -1)), BALL_SPEED * random.choice((1, -1))]
        self.score = 0
        self.frame_iteration = 0
        self.previous_y_distance = 0
        self.previous_ai_y = self.ai_paddle.centery
        self._update_intelligence_level()
        state = self._get_state()
        logging.debug(f"Reset state: {state}, shape: {state.shape}")
        return state

    def _get_state(self):
        ai_y = (self.ai_paddle.centery / WINDOW_HEIGHT) * 2 - 1
        simple_y = (self.simple_paddle.centery / WINDOW_HEIGHT) * 2 - 1
        ball_x = (self.ball.centerx / WINDOW_WIDTH) * 2 - 1
        ball_y = (self.ball.centery / WINDOW_HEIGHT) * 2 - 1
        ball_dx, ball_dy = self.ball_speed[0] / BALL_SPEED, self.ball_speed[1] / BALL_SPEED
        y_distance = ball_y - ai_y
        relative_velocity_y = ball_dy - (self.ai_paddle.centery - self.previous_ai_y) / WINDOW_HEIGHT
        state = np.array([ai_y, simple_y, ball_x, ball_y, ball_dx, ball_dy, y_distance, relative_velocity_y], dtype=float)
        logging.debug(f"State: {state}, shape: {state.shape}, AI paddle y: {self.ai_paddle.y}")
        return state

    def play_step(self, action):
        if not self.paused and not self.showing_progress:
            self.frame_iteration += 1
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if self._is_button_clicked(event.pos, 'pause'):
                        self._show_training_progress()
                    elif self._is_button_clicked(event.pos, 'load'):
                        self._load_saved_data()

            self.previous_ai_y = self.ai_paddle.centery
            logging.debug(f"Action chosen: {action}, AI paddle y before: {self.ai_paddle.y}")
            if action == 0:
                self.ai_paddle.y -= PADDLE_SPEED
            elif action == 1:
                self.ai_paddle.y += PADDLE_SPEED
            self.ai_paddle.clamp_ip(self.display.get_rect())
            logging.debug(f"AI paddle y after: {self.ai_paddle.y}")

            target_y = self.ball.centery + self.ball_speed[1] * 8
            if self.simple_paddle.centery < target_y:
                self.simple_paddle.y += PADDLE_SPEED_OPPONENT
            elif self.simple_paddle.centery > target_y:
                self.simple_paddle.y -= PADDLE_SPEED_OPPONENT
            self.simple_paddle.clamp_ip(self.display.get_rect())

            self.ball.x += self.ball_speed[0]
            self.ball.y += self.ball_speed[1]

            reward = 0
            game_over = False
            ai_y = (self.ai_paddle.centery / WINDOW_HEIGHT) * 2 - 1
            ball_y = (self.ball.centery / WINDOW_HEIGHT) * 2 - 1
            current_y_distance = abs(ball_y - ai_y)
            logging.debug(f"Current y_distance: {current_y_distance}")
            if current_y_distance < 0.1:
                reward += 2.0 / (1 + current_y_distance)
            if current_y_distance < self.previous_y_distance:
                reward += 0.5 * (self.previous_y_distance - current_y_distance)
            if action == 0 and ball_y > ai_y:
                reward -= 0.3
            elif action == 1 and ball_y < ai_y:
                reward -= 0.3
            if self.ball.colliderect(self.ai_paddle):
                self.ball_speed[0] = -self.ball_speed[0]
                self.ball.x = self.ai_paddle.right
                reward += 10
                self.score += 1
            elif self.ball.colliderect(self.simple_paddle):
                self.ball_speed[0] = -self.ball_speed[0]
                self.ball.x = self.simple_paddle.left - BALL_SIZE
                reward -= 1
            if self.ball.top <= 0:
                self.ball_speed[1] = -self.ball_speed[1]
                self.ball.y = 0
            elif self.ball.bottom >= WINDOW_HEIGHT:
                self.ball_speed[1] = -self.ball_speed[1]
                self.ball.y = WINDOW_HEIGHT - BALL_SIZE
            if self.ball.left <= 0:
                reward = -10
                game_over = True
            elif self.ball.right >= WINDOW_WIDTH:
                reward = 5
                game_over = True
            if self.frame_iteration > 1000:
                game_over = True
                reward = -5

            self.previous_y_distance = current_y_distance
            logging.debug(f"Reward: {reward}, Score: {self.score}")

            if game_over:
                self.scores.append(self.score)
                logging.info(f"Episode ended, Score: {self.score}, Game count: {self.game_count}")
                return reward, game_over, self.score, self.game_count

            self._update_ui()
            pygame.time.Clock().tick(60)
            return reward, game_over, self.score, self.game_count
        return 0, False, self.score, self.game_count

    def _update_ui(self):
        self.display.fill(BLACK)
        pygame.draw.rect(self.display, AI_COLOR, self.ai_paddle)
        pygame.draw.rect(self.display, OPPONENT_COLOR, self.simple_paddle)
        pygame.draw.ellipse(self.display, WHITE, self.ball)
        pygame.draw.aaline(self.display, WHITE, (WINDOW_WIDTH // 2, 0), (WINDOW_WIDTH // 2, WINDOW_HEIGHT))

        score_text = self.font.render(f'Score: {self.score}', True, WHITE)
        game_count_text = self.font.render(f'Game: {self.game_count}', True, WHITE)
        intelligence_text = self.font.render(f'Intelligence: {self.intelligence_level}', True, WHITE)
        epsilon_text = self.font.render(f'Epsilon: {self.agent.epsilon:.3f}', True, WHITE)
        y_distance_text = self.font.render(f'Y-Distance: {self.previous_y_distance:.3f}', True, WHITE)
        self.display.blit(score_text, (10, 10))
        self.display.blit(game_count_text, (10, 30))
        self.display.blit(intelligence_text, (10, 50))
        self.display.blit(epsilon_text, (10, 70))
        self.display.blit(y_distance_text, (10, 90))

        pause_button_rect = pygame.Rect(WINDOW_WIDTH - 70, 10, 60, 30)
        pause_color = BUTTON_HOVER_COLOR if pause_button_rect.collidepoint(pygame.mouse.get_pos()) else BUTTON_COLOR
        pygame.draw.rect(self.display, pause_color, pause_button_rect)
        pause_text = self.button_font.render('Pause', True, WHITE)
        self.display.blit(pause_text, (WINDOW_WIDTH - 65, 15))

        load_button_rect = pygame.Rect(WINDOW_WIDTH - 70, 50, 60, 30)
        load_color = BUTTON_HOVER_COLOR if load_button_rect.collidepoint(pygame.mouse.get_pos()) else BUTTON_COLOR
        pygame.draw.rect(self.display, load_color, load_button_rect)
        load_text = self.button_font.render('Load', True, WHITE)
        self.display.blit(load_text, (WINDOW_WIDTH - 65, 55))

        pygame.display.flip()

    def _is_button_clicked(self, pos, button_type):
        if button_type == 'pause':
            return pygame.Rect(WINDOW_WIDTH - 70, 10, 60, 30).collidepoint(pos)
        elif button_type == 'load':
            return pygame.Rect(WINDOW_WIDTH - 70, 50, 60, 30).collidepoint(pos)
        return False

    def _show_training_progress(self):
        self.showing_progress = True
        self.paused = True
        plt.figure(figsize=(10, 6))
        plt.plot(self.scores, label='Raw Scores', color='purple', alpha=0.5)
        if len(self.scores) >= 100:
            window = np.ones(100) / 100
            moving_avg = np.convolve(self.scores, window, mode='valid')
            plt.plot(range(99, len(self.scores)), moving_avg, label='100-Episode Running Average', color='red', linewidth=2)
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True)
        plt.savefig('training_progress.png')
        plt.show()
        self.showing_progress = False
        self.paused = False

    def _update_intelligence_level(self):
        if len(self.scores) > 0:
            moving_avg = np.mean(self.scores[-min(10, len(self.scores)):])
            base_level = self.game_count // 10
            score_bonus = max(0, moving_avg // 2)
            self.intelligence_level = max(1, base_level + score_bonus)
        else:
            self.intelligence_level = 1

    def _load_saved_data(self):
        self.paused = True
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(initialdir="models", filetypes=[("Model files", "*.pth"), ("All files", "*.*")])
        if file_path:
            try:
                self.agent.model.load_state_dict(torch.load(file_path))
                self.agent.target_model.load_state_dict(torch.load(file_path))
                episode_num = file_path.split('model_episode_')[-1].replace('.pth', '') if 'model_episode_' in file_path else 'best'
                scores_file = f"scores_episode_{episode_num}.npy"
                scores_path = os.path.join('models', scores_file)
                if os.path.exists(scores_path):
                    self.scores = list(np.load(scores_path))
                    self.game_count = len(self.scores) + 1
                    self._update_intelligence_level()
                else:
                    print(f"Warning: No corresponding scores file found at {scores_path}")
                print(f"Loaded model and scores from {file_path}")
            except Exception as e:
                print(f"Error loading file: {e}")
        root.destroy()
        self.paused = False

# Hàm chính
def main():
    if not os.path.exists('models'):
        os.makedirs('models')

    state_size = 8
    action_size = 3
    agent = Agent(state_size, action_size)
    
    file_path = select_model_file()
    if file_path and os.path.exists(file_path):
        try:
            agent.model.load_state_dict(torch.load(file_path))
            agent.target_model.load_state_dict(torch.load(file_path))
            logging.info(f"Loaded model from {file_path}")
            print(f"Đã load mô hình từ {file_path}")
        except Exception as e:
            logging.error(f"Failed to load model from {file_path}: {e}")
            print(f"Không thể load mô hình từ {file_path}. Bắt đầu với mô hình mới.")
    else:
        logging.info("No model file selected or file not found, starting with a new model")
        print("Không có file mô hình được chọn, bắt đầu với mô hình mới.")

    game = PongGame(agent)
    batch_size = 32
    max_episodes = 10000
    best_score = float('-inf')
    replay_frequency = 4

    for episode in range(max_episodes):
        state = game.reset()
        done = False
        step = 0
        while not done:
            action = agent.act(state)
            reward, done, score, game_count = game.play_step(action)
            next_state = game._get_state()
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(agent.device)
                q_current = agent.model(state_tensor)[0][action].item()
                q_next = torch.max(agent.target_model(next_state_tensor), dim=1)[0].item()
                error = reward + agent.gamma * q_next * (1 - done) - q_current
            agent.remember(state, action, reward, next_state, done, error)
            state = next_state
            step += 1
            if step % replay_frequency == 0:
                agent.replay(batch_size)
            if done:
                game.scores.append(score)
                if score > best_score:
                    best_score = score
                    torch.save(agent.model.state_dict(), 'models/dqn_model_best.pth')
                    logging.info(f"New best score: {best_score}. Model saved.")
                if episode % 5 == 0:
                    torch.save(agent.model.state_dict(), f'models/dqn_model_episode_{episode}.pth')
                    np.save(f'models/scores_episode_{episode}.npy', np.array(game.scores))
                    logging.info(f"Model and scores saved at episode {episode}")
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
                    plt.close()
                print(f"Episode {episode} - Score: {score}, Steps: {game.frame_iteration}, Epsilon: {agent.epsilon:.3f}")
                break

    pygame.quit()
    logging.info("Training completed")

if __name__ == "__main__":
    main()