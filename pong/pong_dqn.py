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

# Thiết lập logging
logging.basicConfig(filename='snake_training.log', level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Khởi tạo Pygame
pygame.init()

# Cài đặt trò chơi
GRID_SIZE = 20  # Kích thước ô lưới
GRID_WIDTH = 20  # Số ô theo chiều ngang
GRID_HEIGHT = 20  # Số ô theo chiều dọc
WINDOW_WIDTH = GRID_WIDTH * GRID_SIZE
WINDOW_HEIGHT = GRID_HEIGHT * GRID_SIZE
SNAKE_SPEED = 10  # FPS cho tốc độ di chuyển

# Màu sắc
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
SNAKE_COLOR = (0, 255, 0)
FOOD_COLOR = (255, 0, 0)
BUTTON_COLOR = (0, 100, 0)
BUTTON_HOVER_COLOR = (0, 150, 0)

# Lớp DQN (giữ nguyên từ mã Pong)
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

# Lớp Agent (giữ nguyên từ mã Pong)
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

# Hàm chọn file mô hình (giữ nguyên từ mã Pong)
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

# Lớp trò chơi Snake
class SnakeGame:
    def __init__(self, agent):
        self.display = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption('Snake Game with DQN')
        self.font = pygame.font.SysFont('arial', 20)
        self.button_font = pygame.font.SysFont('arial', 15)
        self.agent = agent
        self.game_count = 0
        self.scores = []
        self.paused = False
        self.showing_progress = False
        self.intelligence_level = 1
        self.reset()

    def reset(self):
        self.game_count += 1
        self.snake = [(GRID_WIDTH // 2, GRID_HEIGHT // 2)]  # Snake starts at center
        self.direction = (0, -1)  # Up
        self.food = self._place_food()
        self.score = 0
        self.frame_iteration = 0
        self._update_intelligence_level()
        state = self._get_state()
        logging.debug(f"Reset state: {state}, shape: {state.shape}")
        return state

    def _place_food(self):
        while True:
            food = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
            if food not in self.snake:
                return food

    def _get_state(self):
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        # Normalize positions to [-1, 1]
        head_x_norm = (head_x / GRID_WIDTH) * 2 - 1
        head_y_norm = (head_y / GRID_HEIGHT) * 2 - 1
        food_x_norm = (food_x / GRID_WIDTH) * 2 - 1
        food_y_norm = (food_y / GRID_HEIGHT) * 2 - 1
        # Direction as one-hot encoding
        dir_up = 1 if self.direction == (0, -1) else 0
        dir_down = 1 if self.direction == (0, 1) else 0
        dir_left = 1 if self.direction == (-1, 0) else 0
        dir_right = 1 if self.direction == (1, 0) else 0
        # Danger indicators (wall or body in adjacent cells)
        danger_up = 1 if head_y == 0 or (head_x, head_y - 1) in self.snake else 0
        danger_down = 1 if head_y == GRID_HEIGHT - 1 or (head_x, head_y + 1) in self.snake else 0
        danger_left = 1 if head_x == 0 or (head_x - 1, head_y) in self.snake else 0
        danger_right = 1 if head_x == GRID_WIDTH - 1 or (head_x + 1, head_y) in self.snake else 0
        state = np.array([
            head_x_norm, head_y_norm, food_x_norm, food_y_norm,
            dir_up, dir_down, dir_left, dir_right,
            danger_up, danger_down, danger_left, danger_right
        ], dtype=float)
        logging.debug(f"State: {state}, shape: {state.shape}")
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

            # Map actions: 0=up, 1=down, 2=left, 3=right
            new_direction = [(0, -1), (0, 1), (-1, 0), (1, 0)][action]
            # Prevent reversing direction
            if (new_direction[0] != -self.direction[0] or new_direction[1] != -self.direction[1]):
                self.direction = new_direction

            # Move snake
            head_x, head_y = self.snake[0]
            new_head = (head_x + self.direction[0], head_y + self.direction[1])
            self.snake.insert(0, new_head)

            reward = 0
            game_over = False

            # Check collisions
            if (new_head[0] < 0 or new_head[0] >= GRID_WIDTH or
                new_head[1] < 0 or new_head[1] >= GRID_HEIGHT or
                new_head in self.snake[1:]):
                reward = -10
                game_over = True
            else:
                # Check food collision
                if new_head == self.food:
                    self.score += 1
                    reward = 10
                    self.food = self._place_food()
                else:
                    self.snake.pop()  # Remove tail if no food eaten
                # Small positive reward for surviving
                reward += 0.1

            # Timeout to prevent infinite games
            if self.frame_iteration > 100 * len(self.snake):
                reward = -5
                game_over = True

            if game_over:
                self.scores.append(self.score)
                logging.info(f"Episode ended, Score: {self.score}, Game count: {self.game_count}")
                return reward, game_over, self.score, self.game_count

            self._update_ui()
            pygame.time.Clock().tick(SNAKE_SPEED)
            return reward, game_over, self.score, self.game_count
        return 0, False, self.score, self.game_count

    def _update_ui(self):
        self.display.fill(BLACK)
        # Draw snake
        for segment in self.snake:
            pygame.draw.rect(self.display, SNAKE_COLOR,
                             (segment[0] * GRID_SIZE, segment[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE))
        # Draw food
        pygame.draw.rect(self.display, FOOD_COLOR,
                         (self.food[0] * GRID_SIZE, self.food[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE))
        # Draw grid
        for x in range(0, WINDOW_WIDTH, GRID_SIZE):
            pygame.draw.line(self.display, WHITE, (x, 0), (x, WINDOW_HEIGHT))
        for y in range(0, WINDOW_HEIGHT, GRID_SIZE):
            pygame.draw.line(self.display, WHITE, (0, y), (WINDOW_WIDTH, y))

        score_text = self.font.render(f'Score: {self.score}', True, WHITE)
        game_count_text = self.font.render(f'Game: {self.game_count}', True, WHITE)
        intelligence_text = self.font.render(f'Intelligence: {self.intelligence_level}', True, WHITE)
        epsilon_text = self.font.render(f'Epsilon: {self.agent.epsilon:.3f}', True, WHITE)
        self.display.blit(score_text, (10, 10))
        self.display.blit(game_count_text, (10, 30))
        self.display.blit(intelligence_text, (10, 50))
        self.display.blit(epsilon_text, (10, 70))

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
        plt.savefig('snake_training_progress.png')
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

    state_size = 12  # head_x, head_y, food_x, food_y, dir(4), danger(4)
    action_size = 4  # Up, down, left, right
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

    game = SnakeGame(agent)
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
                    plt.title('Snake Training Progress')
                    plt.legend()
                    plt.grid(True)
                    plt.savefig('snake_training_progress.png')
                    plt.close()
                print(f"Episode {episode} - Score: {score}, Steps: {game.frame_iteration}, Epsilon: {agent.epsilon:.3f}")
                break

    pygame.quit()
    logging.info("Training completed")

if __name__ == "__main__":
    main()