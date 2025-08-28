import random
import numpy as np
import gymnasium as gym
import tensorflow as tf
from collections import deque
import pygame
import os
import logging

# ===========================
# Thiết lập logging
# ===========================
logging.basicConfig(filename='training.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# ===========================
# Tạo môi trường
# ===========================
try:
    import flappy_bird_gym
    env = gym.make("FlappyBird-v0", render_mode="human")
    logging.info("Using FlappyBird-v0 from flappy_bird_gym")
except ImportError as e1:
    logging.error(f"Failed to import flappy_bird_gym: {e1}")
    try:
        import flappy_bird_gymnasium
        env = gym.make("FlappyBird-v0", render_mode="human")
        logging.info("Using FlappyBird-v0 from flappy_bird_gymnasium")
    except ImportError as e2:
        logging.error(f"Failed to import flappy_bird_gymnasium: {e2}")
        logging.info("Fallback to CartPole-v1")
        env = gym.make("CartPole-v1", render_mode="human")

initial_state, _ = env.reset()

# ===========================
# Hàm xử lý state
# ===========================
def preprocess_state(state):
    state = np.array(state, dtype=np.float32)
    if state.ndim > 1:  # nếu là ảnh
        state = state.mean(axis=2) / 255.0  # grayscale + normalize
        state = state.flatten()
    else:  # nếu là vector
        state = np.clip(state, -1.0, 1.0)
    return np.reshape(state, [1, state_size])

# ===========================
# Cấu hình DQN
# ===========================
state_size = initial_state.shape[0] if isinstance(initial_state, np.ndarray) else len(initial_state)
action_size = env.action_space.n
learning_rate = 0.001
memory = deque(maxlen=2000)
gamma = 0.95
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
batch_size = 32

# ===========================
# Xây dựng mô hình DQN
# ===========================
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, input_shape=(state_size,), activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(action_size, activation='linear')
])
model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

# ===========================
# Load weights cũ nếu có
# ===========================
if not os.path.exists("models"):
    os.makedirs("models")

if os.path.exists("models/dqn_latest.weights.h5"):
    model.load_weights("models/dqn_latest.weights.h5")
    logging.info("Loaded previous weights to continue training")

# ===========================
# Bộ nhớ Replay
# ===========================
def remember(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))

# ===========================
# Replay training
# ===========================
def replay():
    if len(memory) < batch_size:
        return

    minibatch = random.sample(memory, batch_size)
    states = np.vstack([m[0] for m in minibatch])
    next_states = np.vstack([m[3] for m in minibatch])

    q_values = model.predict(states, verbose=0)
    q_next = model.predict(next_states, verbose=0)

    for i, (_, action, reward, _, done) in enumerate(minibatch):
        target = reward if done else reward + gamma * np.amax(q_next[i])
        q_values[i][action] = target

    model.fit(states, q_values, epochs=1, verbose=0)

# ===========================
# Chọn hành động
# ===========================
def get_action(state):
    global epsilon
    if np.random.rand() <= epsilon:
        return env.action_space.sample()
    q_values = model.predict(state, verbose=0)[0]
    return np.argmax(q_values)

# ===========================
# Chơi 1 episode
# ===========================
def play_episode(ep):
    global epsilon
    state, _ = env.reset()
    state = preprocess_state(state)
    done = False
    total_reward = 0
    steps = 0

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                logging.info("Game window closed")
                return False, 0

        action = get_action(state)
        next_state, reward, done, truncated, _ = env.step(action)
        next_state = preprocess_state(next_state)

        remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        steps += 1

        epsilon = max(epsilon_min, epsilon * epsilon_decay)  # decay theo step

        if done or truncated:
            break

    logging.info(f"Episode {ep} - Score: {total_reward}, Steps: {steps}")
    print(f"Episode {ep} - Score: {total_reward:.2f}, Steps: {steps}, Epsilon: {epsilon:.3f}")

    replay()
    model.save_weights("models/dqn_latest.weights.h5")  # Lưu sau mỗi lần chết

    return True, total_reward

# ===========================
# Training loop
# ===========================
max_episodes = 1000
best_score = float('-inf')

for episode in range(1, max_episodes + 1):
    playing, score = play_episode(episode)
    if not playing:
        break

    # Lưu best model
    if score > best_score:
        best_score = score
        model.save("models/dqn_model_best.keras")
        logging.info(f"New best score: {best_score}. Model saved.")

    # Lưu model định kỳ
    if episode % 10 == 0:
        model.save(f"models/dqn_model_episode_{episode}.keras")
        logging.info(f"Model saved at episode {episode}")

env.close()
logging.info("Training completed")
