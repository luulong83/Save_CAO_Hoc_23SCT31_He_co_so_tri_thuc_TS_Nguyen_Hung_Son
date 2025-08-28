import pygame
import random
import numpy as np
from collections import deque
import os
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import torch

# Initialize Pygame
pygame.init()

# Game settings
BLOCK_SIZE = 20
GRID_SIZE = 20
WINDOW_SIZE = GRID_SIZE * BLOCK_SIZE
SPEED = 20

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
BUTTON_COLOR = (0, 100, 0)
BUTTON_HOVER_COLOR = (0, 150, 0)
MAGENTA = (255, 0, 255)  # Thay màu vàng bằng màu hồng tím để dễ nhận diện

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

class SnakeGame:
    def __init__(self, agent):
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
                if event.type == pygame.MOUSEBUTTONDOWN:
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
        # Vẽ thân rắn bằng màu xanh lá (tất cả trừ phần đầu)
        if len(self.snake) > 1:  # Đảm bảo có thân để vẽ
            for pt in self.snake[1:]:
                pygame.draw.rect(self.display, GREEN, pygame.Rect(pt.x * BLOCK_SIZE, pt.y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
        # Vẽ đầu rắn bằng màu hồng tím
        if self.snake:  # Đảm bảo self.snake không rỗng
            print("Drawing head in magenta at", self.snake[0].x, self.snake[0].y)  # Debug với màu magenta
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