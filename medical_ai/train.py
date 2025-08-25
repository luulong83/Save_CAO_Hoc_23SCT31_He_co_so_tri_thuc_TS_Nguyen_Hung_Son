import time
import os
import json
from model import train_model

def train_with_progress(data_dir='chest_xray', epochs=10, batch_size=5, learning_rate=0.0005):
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
        "train_accuracy": [],
        "val_accuracy": [],
        "duration_minutes": None
    }

    print("Gọi train_model...")
    history = train_model(
        data_dir=data_dir,
        epochs=epochs,
        batch_size=batch_size,
        progress_bar=True,
        learning_rate=learning_rate
    )
    print("train_model hoàn thành:", history)

    # Lưu train_loss, val_loss, train_accuracy, val_accuracy
    log_data["train_loss"] = history.get('train_loss', [])
    log_data["val_loss"] = history.get('val_loss', [])
    log_data["train_accuracy"] = history.get('train_accuracy', [])
    log_data["val_accuracy"] = history.get('val_accuracy', [])

    end_time = time.time()
    log_data["duration_minutes"] = (end_time - start_time) / 60
    print(f"Hoàn thành train! Tổng thời gian: {log_data['duration_minutes']:.2f} phút")
    print("Log data:", log_data)

    # Lưu log vào file
    with open(log_file, 'a', encoding='utf-8') as f:
        json.dump(log_data, f, ensure_ascii=False)
        f.write("\n")

if __name__ == "__main__":
    train_with_progress(
        data_dir='chest_xray',
        epochs=10,
        batch_size=5,
        learning_rate=0.0005
    )