import logging
import time
import numpy as np
import os
import matplotlib.pyplot as plt
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_XNNPACK_ENABLED'] = '0'
import tensorflow as tf
from tensorflow.keras import Sequential, layers, optimizers
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('khulong_log.txt', mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"XNNPACK disabled: {os.environ.get('TF_XNNPACK_ENABLED')}")
logger.info(f"oneDNN disabled: {os.environ.get('TF_ENABLE_ONEDNN_OPTS')}")
logger.info(f"TensorFlow version: {tf.__version__}")

# ===============================
# Hàm lưu và vẽ Training Progress
# ===============================
def save_training_progress(rewards, filename="training_progress.txt"):
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write("Episode\tReward\n")
            for i, reward in enumerate(rewards):
                f.write(f"{i+1}\t{reward}\n")
        logger.info(f"Đã lưu tiến trình huấn luyện vào {filename}")
    except Exception as e:
        logger.error(f"Lỗi khi lưu tiến trình huấn luyện: {e}")

def plot_training_progress(rewards, filename="training_progress.png"):
    try:
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(rewards) + 1), rewards, label="Reward per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Training Progress")
        plt.legend()
        plt.grid(True)
        plt.savefig(filename)
        plt.close()
        logger.info(f"Đã lưu biểu đồ tiến trình huấn luyện vào {filename}")
    except Exception as e:
        logger.error(f"Lỗi khi vẽ biểu đồ: {e}")

# ===============================
# Thiết lập Selenium
# ===============================
offline_path = r"C:\Users\HP\Documents\HE\khunglong\t-rex-runner-gh-pages\index.html"
chrome_options = Options()
chrome_options.add_argument("--disable-infobars")
chrome_options.add_argument("--disable-extensions")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-notifications")
chrome_options.add_argument("--disable-background-networking")
chrome_options.add_experimental_option("detach", True)
chrome_options.add_argument("--disable-dev-shm-usage")
service = Service(r"C:\chromedriver-win64\chromedriver.exe")
driver = webdriver.Chrome(service=service, options=chrome_options)
driver.set_window_size(1280, 720)
driver.execute_script("document.body.style.zoom='100%'")
driver.get(f"file:///{offline_path}")
wait = WebDriverWait(driver, 10)
try:
    wait.until(lambda d: d.execute_script("return typeof Runner !== 'undefined';"))
    logger.info("Runner khởi tạo thành công")
    logger.info(f"Runner instance: {driver.execute_script('return Runner.instance_;')}")
except Exception as e:
    logger.error(f"Lỗi khởi tạo Runner: {e}")
    driver.quit()
    exit(1)

# ===============================
# Môi trường Dino
# ===============================
class DinoEnv:
    def check_trex_visible(self):
        try:
            t_rex_pos = driver.execute_script("return Runner.instance_.tRex.xPos;")
            t_rex_y = driver.execute_script("return Runner.instance_.tRex.yPos || null;")
            logger.info(f"T-Rex position: x={t_rex_pos}, yPos: {t_rex_y}")
            return t_rex_pos is not None and t_rex_y is not None
        except Exception as e:
            logger.error(f"Lỗi kiểm tra T-Rex: {e}")
            return False

    def reset(self):
        try:
            logger.info("Bắt đầu reset trò chơi")
            driver.execute_script("Runner.instance_.restart();")
            time.sleep(0.1)
            body = driver.find_element(By.TAG_NAME, "body")
            body.send_keys(Keys.SPACE)
            logger.info("Simulated space key to start game")
            time.sleep(1.0)
            playing = driver.execute_script("return Runner.instance_.playing || false;")
            t_rex_status = driver.execute_script("return Runner.instance_.tRex.status;")
            t_rex_pos = driver.execute_script("return Runner.instance_.tRex.xPos || null;")
            t_rex_y = driver.execute_script("return Runner.instance_.tRex.yPos || null;")
            canvas_exists = driver.execute_script("return !!document.querySelector('canvas.runner-canvas');")
            sprite_loaded = driver.execute_script("return !!document.querySelector('img[src*=\"offline-sprite-2x.png\"]');")
            zoom_level = driver.execute_script("return window.devicePixelRatio;")
            if not sprite_loaded:
                sprite_error = driver.execute_script("""
                    try {
                        let img = new Image();
                        img.src = 'offline-sprite-2x.png';
                        img.onerror = () => { return 'Failed to load sprite'; };
                        return img.complete ? 'Sprite exists' : 'Sprite not found';
                    } catch (e) {
                        return 'Error loading sprite: ' + e.message;
                    }
                """)
                logger.info(f"Sprite status: {sprite_error}")
            logger.info(f"Game playing after reset: {playing}, T-Rex status: {t_rex_status}, T-Rex position: x={t_rex_pos}, yPos: {t_rex_y}, Canvas exists: {canvas_exists}, Sprite loaded: {sprite_loaded}, Zoom level: {zoom_level * 100}%")
            if not playing:
                body.send_keys(Keys.SPACE)
                time.sleep(1.0)
                logger.info("Thử chạy lại trò chơi vì Game playing: False")
            if not self.check_trex_visible():
                logger.warning("T-Rex không hiển thị, thử vẽ lại...")
                driver.execute_script("Runner.instance_.tRex.draw(0, 0);")
                driver.execute_script("Runner.instance_.update();")
            state = self.get_state()
            logger.info(f"Trạng thái sau reset: {state}")
            return state
        except Exception as e:
            logger.error(f"Lỗi trong reset: {e}")
            return np.array([0.62, 0, 1, 0.4], dtype=np.float32)

    def get_state(self):
        js = """
        try {
            if (!Runner.instance_ || !Runner.instance_.tRex) return [100, 0, 600, 20];
            let dino = Runner.instance_.tRex;
            let obs = Runner.instance_.horizon.obstacles[0];
            if (!obs) return [dino.yPos || 100, dino.jumpVelocity || 0, 600, 20];
            return [dino.yPos || 100, dino.jumpVelocity || 0, obs.xPos - dino.xPos, obs.width];
        } catch (e) {
            return [100, 0, 600, 20];
        }
        """
        try:
            state = driver.execute_script(js)
            if not isinstance(state, list) or len(state) != 4:
                logger.error(f"Trạng thái không hợp lệ từ JS: {state}")
                return np.array([0.62, 0, 1, 0.4], dtype=np.float32)
            state = np.array(state, dtype=np.float32)
            state[0] = state[0] / 150.0
            state[1] = state[1] / 20.0
            state[2] = state[2] / 600.0
            state[3] = state[3] / 50.0
            if np.any(np.isnan(state)) or np.any(np.isinf(state)):
                logger.error(f"Trạng thái không hợp lệ: {state}")
                return np.array([0.62, 0, 1, 0.4], dtype=np.float32)
            assert state.shape == (4,), f"Kỳ vọng trạng thái (4,), nhận được {state.shape}"
            logger.info(f"Trạng thái: {state}")
            return state
        except Exception as e:
            logger.error(f"Lỗi trong get_state: {e}")
            return np.array([0.62, 0, 1, 0.4], dtype=np.float32)

    def step(self, action):
        try:
            driver.current_window_handle
            if action == 1:
                body = driver.find_element(By.TAG_NAME, "body")
                body.send_keys(Keys.SPACE)
                logger.info("Nhảy được gọi by simulating space key")
            game_speed = driver.execute_script("return Runner.instance_.currentSpeed || 6;")
            playing = driver.execute_script("return Runner.instance_.playing || false;")
            t_rex_y = driver.execute_script("return Runner.instance_.tRex.yPos || null;")
            logger.info(f"Game playing: {playing}, Speed: {game_speed}, T-Rex yPos: {t_rex_y}")
            time.sleep(max(0.005, 0.1 / game_speed))
            state = self.get_state()
            done = driver.execute_script("return Runner.instance_.crashed || false;")
            reward = 1.0 if not done else -10.0
            logger.info(f"Step completed: Action={action}, Done={done}, State={state}")
            return state, reward, done
        except Exception as e:
            logger.error(f"Lỗi trong step: {e}")
            return np.array([0.62, 0, 1, 0.4], dtype=np.float32), -10.0, True

# ===============================
# Tác tử Policy Gradient
# ===============================
class PGAgent:
    def __init__(self, state_size=4, action_size=2, lr=0.001, gamma=0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.states, self.actions, self.rewards = [], [], []
        self.model = Sequential([
            layers.Input(shape=(state_size,)),
            layers.Dense(24, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            layers.Dense(24, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            layers.Dense(action_size, activation='softmax')
        ])
        self.optimizer = optimizers.Adam(learning_rate=lr)
        logger.info("Khởi tạo PGAgent thành công")
        self.load_model("dino_model.h5")

    def choose_action(self, state):
        state = state.reshape([1, self.state_size])
        prob = self.model(state).numpy()[0]
        prob = np.clip(prob, 1e-8, 1.0)
        prob = prob / np.sum(prob)
        if np.any(np.isnan(prob)):
            logger.error(f"NaN trong xác suất: {prob}")
            return np.random.randint(self.action_size)
        action = np.random.choice(self.action_size, p=prob)
        logger.info(f"Chọn hành động: {action}, Xác suất: {prob}")
        return action

    def store(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        logger.info(f"Lưu trạng thái: {state}, Hành động: {action}, Phần thưởng: {reward}")

    def learn(self):
        G = np.zeros_like(self.rewards, dtype=np.float32)
        running_add = 0
        for t in reversed(range(len(self.rewards))):
            running_add = running_add * self.gamma + self.rewards[t]
            G[t] = running_add
        G = (G - np.mean(G)) / (np.std(G) + 1e-8)
        logger.info(f"Phần thưởng chuẩn hóa: {G[:5]}")
        with tf.GradientTape() as tape:
            loss = 0
            entropy = 0
            for s, a, g in zip(self.states, self.actions, G):
                s = s.reshape([1, self.state_size])
                prob = self.model(s)
                loss += -g * tf.math.log(prob[0, a] + 1e-8)
                entropy += -tf.reduce_sum(prob * tf.math.log(prob + 1e-8))
            loss = loss - 0.01 * entropy
            logger.info(f"Loss: {loss.numpy()}, Entropy: {entropy.numpy()}")
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        self.states, self.actions, self.rewards = [], [], []
        logger.info("Học xong, xóa bộ nhớ")

    def save_model(self, path):
        try:
            self.model.save(path)
            logger.info(f"Mô hình được lưu tại: {path}")
        except Exception as e:
            logger.error(f"Lỗi khi lưu mô hình: {e}")

    def load_model(self, path):
        if os.path.exists(path):
            try:
                self.model = tf.keras.models.load_model(path)
                logger.info(f"Mô hình được tải từ: {path}")
            except Exception as e:
                logger.error(f"Lỗi khi tải mô hình: {e}")
        else:
            logger.info(f"Không tìm thấy mô hình tại {path}, sử dụng mô hình mới")

# ===============================
# Vòng lặp huấn luyện
# ===============================
def restart_browser():
    global driver
    try:
        driver.quit()
        logger.info("Đóng trình duyệt cũ")
    except:
        pass
    driver = webdriver.Chrome(service=service, options=chrome_options)
    driver.set_window_size(1280, 720)
    driver.execute_script("document.body.style.zoom='100%'")
    driver.get(f"file:///{offline_path}")
    wait.until(lambda d: d.execute_script("return typeof Runner !== 'undefined';"))
    logger.info("Runner khởi tạo lại thành công")

# Tải dữ liệu rewards nếu tồn tại
rewards = []
if os.path.exists("rewards.npy"):
    try:
        rewards = np.load("rewards.npy").tolist()
        logger.info(f"Đã tải phần thưởng từ rewards.npy: {len(rewards)} episodes")
    except Exception as e:
        logger.error(f"Lỗi khi tải rewards.npy: {e}")
        rewards = []
else:
    logger.info("Không tìm thấy file rewards.npy, khởi tạo danh sách rỗng")

env = DinoEnv()
agent = PGAgent()
episodes = 500
try:
    for ep in range(episodes):
        try:
            driver.current_window_handle
        except:
            logger.warning("Trình duyệt đã đóng, khởi động lại...")
            restart_browser()
        state = env.reset()
        total_reward = 0
        step_count = 0
        max_steps = 500
        logger.info(f"Bắt đầu episode {ep+1}")
        while step_count < max_steps:
            action = agent.choose_action(state)
            logger.info(f"Step {step_count}, State: {state}, Action: {action}")
            next_state, reward, done = env.step(action)
            agent.store(state, action, reward)
            state = next_state
            total_reward += reward
            step_count += 1
            if done:
                agent.learn()
                agent.save_model("dino_model.h5")
                logger.info(f"Episode {ep+1}, Reward: {total_reward}, Steps: {step_count}")
                rewards.append(total_reward)
                save_training_progress(rewards)  # Lưu tiến trình sau mỗi episode
                plot_training_progress(rewards)  # Vẽ biểu đồ sau mỗi episode
                break
        else:
            logger.info(f"Episode {ep+1} vượt quá {max_steps} bước, dừng.")
            agent.learn()
            agent.save_model("dino_model.h5")
            rewards.append(total_reward)
            save_training_progress(rewards)  # Lưu tiến trình
            plot_training_progress(rewards)  # Vẽ biểu đồ
except Exception as e:
    logger.error(f"Lỗi trong vòng lặp huấn luyện: {e}")
finally:
    driver.quit()
    logger.info("Đóng trình duyệt cuối cùng")
    np.save("rewards.npy", rewards)
    agent.save_model("dino_model.h5")
    save_training_progress(rewards)  # Lưu tiến trình cuối cùng
    plot_training_progress(rewards)  # Vẽ biểu đồ cuối cùng