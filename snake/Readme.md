snake_rl/
├─ requirements.txt        # Danh sách các thư viện cần thiết (numpy, pygame, torch, etc.)
├─ snake_env.py           # Môi trường Snake với Pygame render (real-time)
├─ dqn_agent.py           # Triển khai DQN agent và logic học
├─ train_dqn.py           # Script để train DQN và lưu model/scores vào thư mục models
├─ play_dqn.py            # Chạy agent đã train với model đã lưu (render)
├─ play_random.py         # Chạy agent ngẫu nhiên (render)
└─ models/                # Thư mục lưu model (.pth) và scores (.npy)


Lợi ích của Cấu trúc này

Tách biệt trách nhiệm (Separation of Concerns):

snake_env.py: Chỉ chứa logic môi trường (game Snake), bao gồm render với Pygame.
dqn_agent.py: Tập trung vào triển khai thuật toán DQN (neural network, experience replay, etc.).
train_dqn.py: Quản lý quá trình huấn luyện và lưu trữ dữ liệu.
play_dqn.py và play_random.py: Tách biệt các kịch bản chơi, giúp dễ dàng thử nghiệm hoặc mở rộng.


Dựa trên cấu trúc dự án `snake_rl` mà chúng ta đã xây dựng, thứ tự chạy các file phụ thuộc vào mục đích sử dụng (huấn luyện mô hình, chơi với mô hình đã huấn luyện, hoặc chơi ngẫu nhiên). Dưới đây là thứ tự chạy các file phù hợp với từng kịch bản:

### 1. Chuẩn bị Môi trường
Trước khi chạy bất kỳ file nào, bạn cần đảm bảo môi trường virtual đã được kích hoạt và các thư viện đã được cài đặt:
- **Kích hoạt môi trường virtual**:
  ```powershell
  .\myenv\Scripts\Activate.ps1
  ```
- **Di chuyển vào thư mục dự án**:
  ```powershell
  cd C:\Users\HP\Documents\Save_CAO_Hoc_23SCT31_He_co_so_tri_thuc_TS_Nguyen_Hung_Son\ReinforcementLearning\snake_rl
  ```
- **Cài đặt thư viện (nếu chưa cài)**:
  ```powershell
  pip install -r requirements.txt
  ```

### 2. Thứ tự Chạy File theo Kịch bản

#### Kịch bản 1: Huấn luyện Mô hình DQN
Nếu bạn muốn huấn luyện một mô hình mới từ đầu:
1. **Chạy `train_dqn.py`**:
   - Lệnh:
     ```powershell
     python train_dqn.py
     ```
   - Mô tả: File này sẽ khởi tạo môi trường Snake, huấn luyện agent DQN trong 1000 episodes, lưu mô hình và điểm số vào thư mục `models/`, và hiển thị đồ thị tiến trình huấn luyện.
   - Lưu ý: Đảm bảo thư mục `models/` được tạo (sẽ tự động tạo nếu chưa có). Chương trình sẽ mở cửa sổ Pygame để hiển thị game.

#### Kịch bản 2: Chơi với Mô hình Đã Huấn luyện
Nếu bạn đã huấn luyện mô hình và muốn chơi với nó:
1. **Chạy `play_dqn.py`**:
   - Lệnh:
     ```powershell
     python play_dqn.py
     ```
   - Mô tả: File này sẽ tải mô hình đã lưu từ `models/model_best.pth` và chạy agent đã huấn luyện trong môi trường Snake. Bạn có thể sử dụng nút "Load" trong giao diện để tải file mô hình khác nếu cần.
   - Lưu ý: Nếu không tìm thấy `model_best.pth`, chương trình sẽ báo lỗi và yêu cầu huấn luyện hoặc tải file thủ công.

#### Kịch bản 3: Chơi với Agent Ngẫu nhiên
Nếu bạn muốn thử nghiệm agent ngẫu nhiên để so sánh:
1. **Chạy `play_random.py`**:
   - Lệnh:
     ```powershell
     python play_random.py
     ```
   - Mô tả: File này sẽ chạy một agent chọn hành động ngẫu nhiên trong môi trường Snake, hiển thị game qua Pygame.
   - Lưu ý: Không cần mô hình đã huấn luyện, chỉ dùng để kiểm tra hoặc so sánh hiệu suất.

### 3. Thứ tự Đề xuất để Bắt Đầu
Nếu bạn mới bắt đầu và chưa có mô hình nào:
1. **Chạy `train_dqn.py`** để huấn luyện mô hình.
2. Sau khi huấn luyện xong (hoặc trong quá trình huấn luyện), **chạy `play_dqn.py`** để thử agent đã huấn luyện.
3. (Tùy chọn) **Chạy `play_random.py`** để so sánh với hành vi ngẫu nhiên.

### Lưu ý Quan trọng
- **Thứ tự phụ thuộc**: Bạn cần chạy `train_dqn.py` trước nếu muốn dùng `play_dqn.py`, vì `play_dqn.py` phụ thuộc vào file mô hình trong `models/`.
- **Debug**: Nếu gặp lỗi, kiểm tra console output và gửi lại cho tôi. Bạn có thể thêm `print` trong các file để theo dõi tiến trình.
- **Thời gian**: Huấn luyện 1000 episodes có thể mất thời gian (tùy cấu hình máy). Bạn có thể giảm `episodes` trong `train_dqn.py` (e.g., `episodes = 10`) để kiểm tra nhanh.

### Tóm tắt Lệnh
- Huấn luyện: `python train_dqn.py`
- Chơi với mô hình: `python play_dqn.py`
- Chơi ngẫu nhiên: `python play_random.py`

Hãy thử chạy theo thứ tự trên và cho tôi biết kết quả hoặc nếu cần hỗ trợ thêm!