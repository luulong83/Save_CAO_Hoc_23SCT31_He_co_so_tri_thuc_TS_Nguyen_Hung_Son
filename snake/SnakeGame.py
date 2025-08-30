import pygame
import random
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Hoặc thử 'Qt5Agg' nếu TkAgg không hoạt động
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import os
import tkinter as tk
from tkinter import filedialog
import platform
import asyncio

# Initialize Pygame
pygame.init()

# Game settings
BLOCK_SIZE = 20
GRID_SIZE = 20
WINDOW_SIZE = GRID_SIZE * BLOCK_SIZE
SPEED = 20
FPS = 60

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
BUTTON_COLOR = (0, 100, 0)
BUTTON_HOVER_COLOR = (0, 150, 0)
MAGENTA = (255, 0, 255)

# Directions
RIGHT = [1, 0, 0, 0]
LEFT = [0, 1, 0, 0]
UP = [0, 0, 1, 0]
DOWN = [0, 0, 0, 1]
STRAIGHT = [1, 0, 0, 0]

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = DQN(state_size, 128, action_size)
        self.target_model = DQN(state_size, 128, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.eye(3)[random.randrange(self.action_size)]
        state = torch.FloatTensor(state).unsqueeze(0)
        act_values = self.model(state)
        return np.eye(3)[torch.argmax(act_values[0]).item()]

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).unsqueeze(0)
            next_state = torch.FloatTensor(next_state).unsqueeze(0)
            action_idx = np.argmax(action)
            target = reward
            if not done:
                target = reward + self.gamma * torch.max(self.target_model(next_state)[0]).item()
            target_f = self.model(state)
            target_f[0][action_idx] = target
            self.optimizer.zero_grad()
            loss = nn.MSELoss()(target_f, self.model(state))
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class SnakeGame:
    def __init__(self, agent=None):
        self.display = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        pygame.display.set_caption('Snake Game with DQN')
        self.font = pygame.font.SysFont('arial', 20)
        self.button_font = pygame.font.SysFont('arial', 15)
        self.game_count = 0
        self.scores = []
        self.paused = False
        self.showing_progress = False
        self.agent = agent
        self.intelligence_level = 1
        self.model_loaded = False
        self.reset()

    def reset(self):
        self.game_count += 1
        self.direction = RIGHT
        self.head = Point(GRID_SIZE // 2, GRID_SIZE // 2)
        self.snake = [self.head]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        self._update_intelligence_level()
        if self.agent and not self.model_loaded:
            model_path = os.path.join('models', 'model_best.pth')
            if os.path.exists(model_path):
                try:
                    self.agent.model.load_state_dict(torch.load(model_path))
                    scores_path = os.path.join('models', 'scores_episode_best.npy')
                    if os.path.exists(scores_path):
                        self.scores = list(np.load(scores_path))
                        self.game_count = len(self.scores) + 1
                        self._update_intelligence_level()
                    print(f"Automatically loaded best model from {model_path}")
                    self.model_loaded = True
                except Exception as e:
                    print(f"Error loading best model: {e}")
        return self._get_state()

    def _place_food(self):
        x = random.randint(0, GRID_SIZE - 1)
        y = random.randint(0, GRID_SIZE - 1)
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def _get_state(self):
        head_x, head_y = self.head.x, self.head.y
        food_x, food_y = self.food.x, self.food.y

        danger = [0, 0, 0]
        directions = [
            (self.direction, 0),
            ([(self.direction[2], self.direction[3], self.direction[0], self.direction[1])[i] for i in [1, 2, 3, 0]], 1),
            ([(self.direction[3], self.direction[2], self.direction[1], self.direction[0])[i] for i in [3, 0, 1, 2]], 2)
        ]

        for dir, idx in directions:
            new_x, new_y = head_x, head_y
            if dir == RIGHT:
                new_x += 1
            elif dir == LEFT:
                new_x -= 1
            elif dir == UP:
                new_y -= 1
            elif dir == DOWN:
                new_y += 1
            if (new_x < 0 or new_x >= GRID_SIZE or new_y < 0 or new_y >= GRID_SIZE or
                Point(new_x, new_y) in self.snake):
                danger[idx] = 1

        food_left = food_x < head_x
        food_right = food_x > head_x
        food_up = food_y < head_y
        food_down = food_y > head_y

        dir_right = self.direction == RIGHT
        dir_left = self.direction == LEFT
        dir_up = self.direction == UP
        dir_down = self.direction == DOWN

        state = [
            danger[0], danger[1], danger[2],
            food_left, food_right, food_up, food_down,
            dir_left, dir_right, dir_up, dir_down
        ]
        return np.array(state, dtype=int)

    def play_step(self, action):
        if not self.paused and not self.showing_progress:
            self.frame_iteration += 1
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                if event.type == pygame.MOUSEBUTTONDOWN and not self.showing_progress:
                    if self._is_button_clicked(event.pos, 'pause'):
                        self._show_training_progress()
                    elif self._is_button_clicked(event.pos, 'load'):
                        self._load_saved_data()

            clock_wise = [RIGHT, DOWN, LEFT, UP]
            idx = clock_wise.index(self.direction)
            if np.array_equal(action, [1, 0, 0]):
                new_dir = clock_wise[idx]
            elif np.array_equal(action, [0, 1, 0]):
                new_dir = clock_wise[(idx + 1) % 4]
            else:
                new_dir = clock_wise[(idx - 1) % 4]
            self.direction = new_dir

            x, y = self.head.x, self.head.y
            if self.direction == RIGHT:
                x += 1
            elif self.direction == LEFT:
                x -= 1
            elif self.direction == UP:
                y -= 1
            elif self.direction == DOWN:
                y += 1
            self.head = Point(x, y)

            game_over = False
            reward = 0
            if (self.head.x < 0 or self.head.x >= GRID_SIZE or
                self.head.y < 0 or self.head.y >= GRID_SIZE or
                self.head in self.snake or
                self.frame_iteration > 100 * len(self.snake)):
                game_over = True
                reward = -10
                return reward, game_over, self.score, self.game_count

            self.snake.insert(0, self.head)
            if self.head == self.food:
                self.score += 1
                reward = 10
                self._place_food()
            else:
                self.snake.pop()

            if self._is_moving_toward_food():
                reward = 1
            else:
                reward = -1

            self._update_ui()
            pygame.time.Clock().tick(SPEED)

            return reward, game_over, self.score, self.game_count
        return 0, False, self.score, self.game_count

    def _is_moving_toward_food(self):
        head_x, head_y = self.head.x, self.head.y
        food_x, food_y = self.food.x, self.food.y
        if self.direction == RIGHT and food_x > head_x:
            return True
        if self.direction == LEFT and food_x < head_x:
            return True
        if self.direction == UP and food_y < head_y:
            return True
        if self.direction == DOWN and food_y > head_y:
            return True
        return False

    def _update_ui(self):
        self.display.fill(BLACK)
        if len(self.snake) > 1:
            for pt in self.snake[1:]:
                pygame.draw.rect(self.display, GREEN, pygame.Rect(pt.x * BLOCK_SIZE, pt.y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
        if self.snake:
            pygame.draw.rect(self.display, MAGENTA, pygame.Rect(self.snake[0].x * BLOCK_SIZE, self.snake[0].y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x * BLOCK_SIZE, self.food.y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
        
        game_count_text = self.font.render(f'Game: {self.game_count}', True, WHITE)
        score_text = self.font.render(f'Score: {self.score}', True, WHITE)
        intelligence_text = self.font.render(f'Intelligence: {self.intelligence_level}', True, WHITE)
        self.display.blit(game_count_text, (10, 10))
        self.display.blit(score_text, (10, 30))
        self.display.blit(intelligence_text, (10, 50))
        
        pause_button_rect = pygame.Rect(WINDOW_SIZE - 70, 10, 60, 30)
        pause_color = BUTTON_HOVER_COLOR if pause_button_rect.collidepoint(pygame.mouse.get_pos()) else BUTTON_COLOR
        pygame.draw.rect(self.display, pause_color, pause_button_rect)
        pause_text = self.button_font.render('Pause', True, WHITE)
        self.display.blit(pause_text, (WINDOW_SIZE - 65, 15))
        
        load_button_rect = pygame.Rect(WINDOW_SIZE - 70, 50, 60, 30)
        load_color = BUTTON_HOVER_COLOR if load_button_rect.collidepoint(pygame.mouse.get_pos()) else BUTTON_COLOR
        pygame.draw.rect(self.display, load_color, load_button_rect)
        load_text = self.button_font.render('Load', True, WHITE)
        self.display.blit(load_text, (WINDOW_SIZE - 65, 55))

        pygame.display.flip()

    def _is_button_clicked(self, pos, button_type):
        if button_type == 'pause':
            return pygame.Rect(WINDOW_SIZE - 70, 10, 60, 30).collidepoint(pos)
        elif button_type == 'load':
            return pygame.Rect(WINDOW_SIZE - 70, 50, 60, 30).collidepoint(pos)
        return False

    def _show_training_progress(self):
        self.showing_progress = True
        self.paused = True
        print(f"Debug: Scores = {self.scores}")
        if not self.scores:
            print("Warning: No scores available to plot!")
            self.showing_progress = False
            self.paused = False
            return
        plt.figure(figsize=(10, 6))
        plt.plot(self.scores, label='Raw Scores', color='purple', alpha=0.5)
        # Tính trung bình động trên số episode tối đa có sẵn
        window_size = min(100, len(self.scores))
        if window_size > 1:
            window = np.ones(window_size) / window_size
            moving_avg = np.convolve(self.scores, window, mode='valid')
            plt.plot(range(window_size - 1, len(self.scores)), moving_avg, label=f'{window_size}-Episode Running Average', color='red', linewidth=2)
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True)
        plt.savefig('training_progress.png')
        plt.show(block=True)  # Chặn luồng chính cho đến khi đóng
        plt.close()
        self.showing_progress = False
        self.paused = False

    def _update_intelligence_level(self):
        if len(self.scores) > 0:
            moving_avg = np.mean(self.scores[-min(10, len(self.scores)):]) if len(self.scores) > 0 else 0
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
                episode_num = file_path.split('model_episode_')[-1].replace('.pth', '')
                scores_file = f"scores_episode_{episode_num}.npy"
                scores_path = os.path.join('models', scores_file)
                if os.path.exists(scores_path):
                    self.scores = list(np.load(scores_path))
                    self.game_count = len(self.scores) + 1
                    self._update_intelligence_level()
                else:
                    print(f"Warning: No corresponding scores file found at {scores_path}. Game count and intelligence not updated.")
                print(f"Loaded model and scores from {file_path}")
            except Exception as e:
                print(f"Error loading file: {e}")
        root.destroy()
        self.paused = False

async def train():
    if not os.path.exists('models'):
        os.makedirs('models')
    print("Models directory created or exists.")
    try:
        agent = DQNAgent(state_size=11, action_size=3)
        game = SnakeGame(agent)
        print("SnakeGame and DQNAgent initialized successfully.")
    except Exception as e:
        print(f"Initialization error: {e}")
        return
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
                    np.save(os.path.join('models', "scores_episode_best.npy"), np.array(game.scores))
                    print(f"New best model saved with score: {max_score}")
                break
            await asyncio.sleep(1.0 / FPS)
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

async def play():
    agent = DQNAgent(state_size=11, action_size=3)
    game = SnakeGame(agent)
    try:
        agent.model.load_state_dict(torch.load(os.path.join('models', "model_best.pth")))
    except FileNotFoundError:
        print("No trained model found. Please train the model first or load a saved file.")
        return
    agent.epsilon = 0.0
    state = game.reset()
    while True:
        action = agent.act(state)
        reward, done, score, game_count = game.play_step(action)
        state = game._get_state()
        if done:
            print(f"Game Over! Game: {game_count}, Score: {score}")
            break
        await asyncio.sleep(1.0 / FPS)
    pygame.quit()

async def main():
    print("Starting Snake Game...")
    await train()
    await play()

if __name__ == "__main__":
    if platform.system() == "Emscripten":
        asyncio.ensure_future(main())
    else:
        asyncio.run(main())