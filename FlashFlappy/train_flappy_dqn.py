import gymnasium as gym
import numpy as np
import tensorflow as tf
from collections import deque
import random
import flappy_bird_gymnasium

# Khởi tạo môi trường
env = gym.make("FlappyBird-v0", render_mode="human")

# Debug: Kiểm tra trạng thái từ reset
initial_state, _ = env.reset()
print("Initial state shape:", initial_state.shape)
print("Initial state:", initial_state)

# Cấu hình DQN (sẽ cập nhật sau khi debug)
state_size = initial_state.shape[0] if len(initial_state.shape) == 1 else initial_state.shape[1] * initial_state.shape[2]  # Điều chỉnh dựa trên shape
action_size = 2  # 0: do nothing, 1: flap
learning_rate = 0.001
memory = deque(maxlen=2000)
gamma = 0.95  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_min = 0.01
epsilon_decay = 0.995
batch_size = 32

# Xây dựng mô hình DQN
model = tf.keras.Sequential([
    tf.keras.layers.Dense(24, input_shape=(state_size,), activation='relu'),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(action_size, activation='linear')
])
model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

# Hàm lấy hành động
def get_action(state):
    if np.random.rand() <= epsilon:
        return env.action_space.sample()
    q_values = model.predict(state)
    return np.argmax(q_values[0])

# Huấn luyện
episodes = 1000
for e in range(episodes):
    state, _ = env.reset()
    # Điều chỉnh shape dựa trên kết quả debug
    if len(state.shape) > 1:
        state = state.flatten()  # Nếu là hình ảnh, làm phẳng
    state = np.reshape(state, [1, state_size])
    done = False
    total_reward = 0

    while not done:
        action = get_action(state)
        next_state, reward, done, _, _ = env.step(action)
        if len(next_state.shape) > 1:
            next_state = next_state.flatten()
        next_state = np.reshape(next_state, [1, state_size])
        total_reward += reward

        # Lưu vào memory
        memory.append((state, action, reward, next_state, done))
        state = next_state

        # Huấn luyện với batch
        if len(memory) > batch_size:
            batch = random.sample(memory, batch_size)
            states = np.array([item[0] for item in batch])
            actions = np.array([item[1] for item in batch])
            rewards = np.array([item[2] for item in batch])
            next_states = np.array([item[3] for item in batch])
            dones = np.array([item[4] for item in batch])

            targets = rewards + gamma * np.max(model.predict(next_states), axis=1) * (1 - dones)
            target_f = model.predict(states)
            for i, action in enumerate(actions):
                target_f[i][action] = targets[i]
            model.fit(states, target_f, epochs=1, verbose=0)

    # Giảm epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    print(f"Episode: {e+1}/{episodes}, Score: {total_reward}, Epsilon: {epsilon:.2f}")

    # Lưu mô hình
    if (e + 1) % 100 == 0:
        model.save(f"flappy_dqn_model_{e+1}.h5")

env.close()