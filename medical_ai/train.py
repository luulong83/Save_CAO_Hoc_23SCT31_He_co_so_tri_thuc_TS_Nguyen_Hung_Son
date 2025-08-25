import time
from model import train_model
from tqdm import tqdm
import os
import json

def train_with_progress(data_dir='chest_xray_balanced', epochs=5, batch_size=8, learning_rate=0.0005):
    print("Bắt đầu quá trình train model...")
    print(f"Dataset: {data_dir}")
    print(f"Số epoch: {epochs}, Batch size: {batch_size}, Learning rate: {learning_rate}")
    start_time = time.time()
    
    # Ghi log huấn luyện
    log_file = 'training_log.json'
    log_data = {
        "dataset": data_dir,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)),
        "train_loss": [],
        "val_loss": [],
        "duration_minutes": None
    }

    # Gọi train_model với progress_bar và truyền learning_rate
    train_model(
        data_dir=data_dir,
        epochs=epochs,
        batch_size=batch_size,
        progress_bar=True,
        learning_rate=learning_rate
    )
    
    end_time = time.time()
    duration_minutes = (end_time - start_time) / 60
    log_data["duration_minutes"] = duration_minutes
    print(f"Hoàn thành train! Tổng thời gian: {duration_minutes:.2f} phút")

    # Lưu log vào file
    with open(log_file, 'a', encoding='utf-8') as f:
        json.dump(log_data, f, ensure_ascii=False)
        f.write("\n")

if __name__ == "__main__":
    train_with_progress(
        data_dir='chest_xray',  # Sử dụng dataset đã cân bằng
        epochs=5,
        batch_size=8,
        learning_rate=0.0005
    )