Hướng Dẫn Chi Tiết Xây Dựng Project Hệ Thống Y Tế Thông Minh Trên PC Bàn (Không GPU)
Dựa trên project trước, tôi sẽ cung cấp hướng dẫn chi tiết từng bước để bạn có thể copy-paste code và chạy thực tế trên máy bàn PC (chạy trên CPU). Project sử dụng Python 3 (tôi giả sử bạn dùng Python 3.8+), PyTorch (chạy CPU), và SWI-Prolog cho phần Symbolic AI. Tôi sẽ tập trung vào việc làm cho code đơn giản, dễ chạy, và sử dụng dataset public nhỏ để tránh tải nặng.
Bước 1: Cài Đặt Môi Trường

Cài Python và Các Thư Viện:

Nếu chưa có Python, tải từ https://www.python.org/downloads/ (chọn Python 3.10 hoặc mới hơn).
Mở Command Prompt (Windows) hoặc Terminal (nếu dùng Linux/Mac), tạo virtual environment:

# python -m venv medical_ai_env

Kích hoạt (Windows: medical_ai_env\Scripts\activate; Linux/Mac: source medical_ai_env/bin/activate).
Cài thư viện cần thiết (chạy trên CPU, không cần CUDA):

# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# pip install pyswip flask pillow

torch: ML framework (phiên bản CPU).
pyswip: Tích hợp Python với Prolog.
flask: Cho web app đơn giản.
pillow: Xử lý hình ảnh.



medical_ai/
├── app.py                # Code tích hợp và Flask app
├── model.py              # Module ML/DL
├── medical_rules.pl      # File Prolog
├── chest_xray/           # Dataset folder
├── trained_model.pth     # Model sau train (sẽ tạo)
└── temp.jpg              # Hình ảnh tạm (khi upload)


# Start Menu → SWI-Prolog
swipl

?- [medical_rules].

# Thoát
?- halt.
